/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/**
 * @file   kvstore_dist.h
 * @brief  distributed implementation based on ps-lite
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_H_
#define MXNET_KVSTORE_KVSTORE_DIST_H_
#include <string>
#include <vector>
#include "./kvstore_local.h"
#include "mxnet/engine.h"
#include "ps/ps.h"
#include "./kvstore_dist_server.h"
#if MKL_EXPERIMENTAL == 1
#include <mkl_memory.h>
#include "../operator/mkl/mkl_memory-inl.h"
#include "../operator/mkl/mkl_util-inl.h"
#endif
namespace mxnet {
namespace kvstore {

/**
 * \brief distributed kvstore
 *
 * for a worker node, it always guarantees that all push and pull issued from
 * this worker on the same key are serialized. namely push(3) and then pull(3),
 * then the data pulled is always containing the modification from the push(3).
 *
 * it's the server node's job to control the data consistency among all
 * workers. see details on \ref ServerHandle::Start
 */
class KVStoreDist : public KVStoreLocal {
 public:
  explicit KVStoreDist(bool use_device_comm)
      : KVStoreLocal(use_device_comm), ps_worker_(nullptr), server_(nullptr) {
    if (IsWorkerNode()) {
      ps_worker_ = new ps::KVWorker<real_t>(0);
      ps::StartAsync("mxnet\0");
      if (!ps::Postoffice::Get()->is_recovery()) {
        ps::Postoffice::Get()->Barrier(
          ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
      }
    }
    bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);
    // TODO: small chuncks for kvstore
    // key_offset_ = 0;

    // // debug
    // LG << "bigarray_bound_=" << bigarray_bound_;

    // // initialization
    // iteration_ = -1;

    const char *partial_pull = ps::Environment::Get()->find("DMLC_PS_PULL_THRESHOLD");
    if (partial_pull == nullptr) {
      partial_pull_ = false;
    }
    else {
      partial_pull_ = true;
    }

    const char *partial_pull_history = ps::Environment::Get()->find("MXNET_KVSTORE_PARTIAL_PULL_HISTORY");
    if (partial_pull_history == nullptr) {
      partial_pull_history_ = false;
    }
    else {
      partial_pull_history_alpha_ = atof(partial_pull_history);
      if (partial_pull_history_alpha_ > 0) {
        partial_pull_history_ = true;
        LG << "Use historical info for partial pulling! alpha=" << partial_pull_history_alpha_;
      }
      else {
        partial_pull_history_ = false;
      }
      
    }

  }

  virtual ~KVStoreDist() {
    Engine::Get()->WaitForAll();
    if (IsWorkerNode()) {
      if (barrier_before_exit_) {
        Barrier();
        if (get_rank() == 0) {
          // stop the executor at servers
          SendCommandToServers(kStopServer, "");
        }
      }
      ps::Finalize(barrier_before_exit_);
      delete ps_worker_;
    }
  }

  void Init(const std::vector<int>& keys,
            const std::vector<NDArray>& values) override {
    CheckUnique(keys);
    for (size_t i = 0; i < keys.size(); ++i) {
      comm_->Init(keys[i], values[i].shape(), values[i].dtype());
    }
    if (get_rank() == 0) {
      Push_(keys, values, 0, false);
      // wait until the push is finished
      for (const auto& v : values) {
        v.WaitToWrite();
      }
    } else {
      // do nothing
    }
    if (!ps::Postoffice::Get()->is_recovery()) {
      Barrier();
    }
  }

  void Push(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority) override {
    // debug
    // LG << "Push called: " << keys[0];
    // LG << "Worker push itr: " << iteration_;
    Push_(keys, values, priority, true);
  }

  void Pull(const std::vector<int>& keys,
            const std::vector<NDArray*>& values,
            int priority) override {

    // debug
    // LG << "Pull called: " << keys[0];

    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray*> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    // // debug
    // std::ostringstream key_list;
    // key_list << "key_list: ";
    // for (size_t i = 0; i < uniq_keys.size(); ++i) {
    //   key_list << uniq_keys[i] << ", ";
    // }
    // LG << key_list.str();

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      int key = uniq_keys[i];

      // initialize
      if (store_iteration_.count(key) == 0) store_iteration_[key] = -1;

      int iteration = store_iteration_[key];

      // use the same array for merging to guarantee that pull always happens
      // after the previous push on this key
      auto& recv_buf = pull_buf_[key];
      auto& pull_prev_buf = pull_prev_buf_[key];
      auto& grad_buf = grad_buf_[key];
      if (recv_buf.is_none()) {
        // it may happen for the first time a no-rank-0 worker pull the weight.
        recv_buf = NDArray(
          grouped_vals[i][0]->shape(), pinned_ctx_, true, grouped_vals[i][0]->dtype());
      }

      if (partial_pull_history_) {
        if (pull_prev_buf.is_none()) {
          pull_prev_buf = NDArray(
            grouped_vals[i][0]->shape(), pinned_ctx_, true, grouped_vals[i][0]->dtype());
        }
        if (grad_buf.is_none()) {
          grad_buf = NDArray(
            grouped_vals[i][0]->shape(), pinned_ctx_, true, grouped_vals[i][0]->dtype());
        }
      }

      if (partial_pull_history_ && iteration > -1) {
        CopyFromTo(recv_buf, &grad_buf);
        if(iteration > 0) {
          recv_buf *= (1+partial_pull_history_alpha_);
          recv_buf -= (pull_prev_buf * partial_pull_history_alpha_);
        }
        CopyFromTo(grad_buf, &pull_prev_buf);
      }

      auto pull_from_servers = [this, key, recv_buf, iteration](
          RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
        size_t size = recv_buf.shape().Size();
        // LG << "EncodeKey in Pull";
        PSKV& pskv = EncodeKey(key, size);
#if MKL_EXPERIMENTAL == 1
        mkl_set_tblob_eager_mode(recv_buf.data());
#endif
        real_t* data = static_cast<real_t*>(recv_buf.data().dptr_);
        // false means not to delete data when SArray is deleted
        auto vals = new ps::SArray<real_t>(data, size, false);
        // issue pull
        // TODO: check thread-safe for iteration counter
        // LG << "MyRank: " << ps::MyRank() << ", pulling: " << key << ", iteration: " << store_iteration_[key];
        CHECK_NOTNULL(ps_worker_)->ZPull(
            pskv.keys, vals, &pskv.lens, 0, [vals, cb](){ delete vals; cb(); }, iteration);
      };

      CHECK_NOTNULL(Engine::Get())->PushAsync(
          pull_from_servers,
          pinned_ctx_,
          {},
          {recv_buf.var()},
          FnProperty::kNormal,
          priority,
          PROFILER_MESSAGE("KVStoreDistPull"));

      store_iteration_[key] = store_iteration_[key] + 1;

      comm_->Broadcast(key, recv_buf, grouped_vals[i], priority);
    }
  }

  void set_updater(const Updater& updater) override {
    CHECK(updater) << "invalid updater";
    // LG << "setting updater";
    if (IsServerNode()) {
      CHECK_NOTNULL(server_)->set_updater(updater);
    } else {
      updater_ = updater;
      // LG << "updater set for worker";
    }
  }

  void Barrier() override {
    ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
  }


  void SendCommandToServers(int cmd_id,
                            const std::string& cmd_body) override {
    CHECK_NOTNULL(ps_worker_);
    ps_worker_->Wait(ps_worker_->Request(cmd_id, cmd_body, ps::kServerGroup));
  }

  int get_group_size() const override { return ps::NumWorkers(); }

  int get_rank() const override { return ps::MyRank(); }

  int get_num_dead_node(int node_id, int timeout) const override {
    int number = 0;
    auto dead_nodes = ps::Postoffice::Get()->GetDeadNodes(timeout);
    const auto& watch_nodes = ps::Postoffice::Get()->GetNodeIDs(node_id);
    std::unordered_set<int> watch_set(watch_nodes.begin(), watch_nodes.end());
    for (int r : dead_nodes) {
      if (watch_set.find(r) != watch_set.end()) number++;
    }
    return number;
  }

  void RunServer(const Controller& controller) override {
    CHECK(!IsWorkerNode());
    if (IsServerNode()) {
      server_ = new KVStoreDistServer();
      server_->set_controller(controller);
    }

    ps::StartAsync("mxnet_server\0");
    if (!ps::Postoffice::Get()->is_recovery()) {
      ps::Postoffice::Get()->Barrier(
        ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
    }
    if (server_) server_->Run();
    ps::Finalize();
    if (server_) {
      delete server_;
    }
    server_ = nullptr;
  }

 private:
  void Push_(const std::vector<int>& keys,
             const std::vector<NDArray>& values,
             int priority,
             bool do_merge)  {
    // first aggregate the values over keys
    std::vector<int> uniq_keys;
    std::vector<std::vector<NDArray> > grouped_vals;
    GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

    for (size_t i = 0; i < uniq_keys.size(); ++i) {
      // merge over devcies
      int key = uniq_keys[i];

      // debug
      // initialize
      // CHECK_NE(store_iteration_.count(key), 0);
      if (store_iteration_.count(key) == 0) store_iteration_[key] = -1;

      const auto& vals = grouped_vals[i];
      NDArray merged = do_merge ? comm_->Reduce(key, vals, priority) : vals[0];

      

      auto& send_buf = grad_buf_[key];
      if (merged.ctx().dev_mask() == cpu::kDevMask) {
        send_buf = merged;  // avoid memory copy
      } else {
        if (send_buf.is_none()) {
          send_buf = NDArray(merged.shape(), pinned_ctx_, true, merged.dtype());
        }
        CopyFromTo(merged, &send_buf);
      }

      int iteration = store_iteration_[key];

      // // debug
      // if (iteration % 10 == 0) LG << ps::MyRank();

      // push to servers
      auto push_to_servers =
          [this, key, send_buf, iteration](RunContext rctx, Engine::CallbackOnComplete cb) {
        // convert to ps keys
        size_t size = send_buf.shape().Size();

        // debug
        // LG << "EncodeKey in Push_";
        PSKV& pskv = EncodeKey(key, size);

        if (iteration == -1) {
          LG << "key: " << key << ", #chunks: " << pskv.keys.size();
        }

#if MKL_EXPERIMENTAL == 1
        mkl_set_tblob_eager_mode(send_buf.data());
#endif
        real_t* data = static_cast<real_t*>(send_buf.data().dptr_);
        // do push. false means no delete
        ps::SArray<real_t> vals(data, size, false);
        CHECK_NOTNULL(ps_worker_)->ZPush(
            pskv.keys, vals, pskv.lens, 0, [cb]() { cb(); }, iteration);
      };
      Engine::Get()->PushAsync(
          push_to_servers,
          pinned_ctx_,
          {send_buf.var()},
          {},
          FnProperty::kNormal,
          priority,
          PROFILER_MESSAGE("KVStoreDistPush"));
    }
  }

  /**
   * \brief check if the keys are all unique
   */
  void CheckUnique(const std::vector<int>& keys) {
    auto keys_copy = keys;
    auto last = std::unique(keys_copy.begin(), keys_copy.end());
    CHECK_EQ(static_cast<size_t>(std::distance(keys_copy.begin(), last)),
             static_cast<size_t>(keys.size()));
  }

  /**
   * \brief struct for ps keys and lens
   */
  struct PSKV {
    ps::SArray<ps::Key> keys;  // n keys
    ps::SArray<int> lens;  // the length of the i-th value
    int size;
  };

  /**
   * \brief cache all key partitions
   */
  std::unordered_map<int, PSKV> ps_kv_;

  /**
   * \brief serizelize EncodeKey
   */
  std::mutex mu_;

  /**
   * \brief convert to keys in ps
   */
  inline PSKV& EncodeKey(int key, size_t size) {
    mu_.lock();
    PSKV& pskv = ps_kv_[key];
    mu_.unlock();

    if (!pskv.keys.empty()) {
      CHECK_EQ(static_cast<size_t>(pskv.size), size) << "The value size cannot be changed";
    } else {
      auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
      int num_servers = krs.size();
      CHECK_GT(num_servers, 0);

      // a simple heuristic for load balance
      if (size < bigarray_bound_) {
        // send it to a single random picked server
        int server = (key * 9973) % num_servers;
        ps::Key ps_key = krs[server].begin() + key;
        CHECK_LT(ps_key, krs[server].end());
        pskv.keys.push_back(ps_key);
        pskv.lens.push_back(size);
        pskv.size = size;
      } else {
        // parition it to all servers
        pskv.size = 0;

        // // debug
        // std::ostringstream key_list;
        // key_list << "key_list: ";
        // CHECK_LE(num_chunks, num_servers);

        for (int i = 0; i < num_servers; ++i) {
          size_t part_size =
              static_cast<size_t>(round(static_cast<double>(size)/num_servers*(i+1))) -
              static_cast<size_t>(round(static_cast<double>(size)/num_servers*i));
          ps::Key ps_key = krs[i].begin() + key;          

          // debug 
          // LG << "key: " << key << " ps_key: " << ps_key << " krs_len: " << krs.size();
          // key_list << ps_key << ", ";

          CHECK_LT(ps_key, krs[i].end());
          pskv.keys.push_back(ps_key);
          pskv.lens.push_back(part_size);
          pskv.size += part_size;
        }

        // debug
        // LG << key_list.str();

        CHECK_EQ(static_cast<size_t>(pskv.size), size);
      }
    }

    // LG << "number of keys: " << pskv.keys.size();

    return pskv;
  }

  /**
   * \brief for worker to push and pull data
   */
  ps::KVWorker<real_t>* ps_worker_;
  /**
   * \brief the server handle
   */
  KVStoreDistServer* server_;
  /**
   * \brief threshold for partition
   */
  size_t bigarray_bound_;
  /// \brief send & recver buffer
  // std::unordered_map<int, NDArray> comm_buf_;
  /**
   * \brief the iteration counter
   */
  // int iteration_;
  std::unordered_map<ps::Key, int> store_iteration_;
  std::unordered_map<ps::Key, NDArray> grad_buf_;
  std::unordered_map<ps::Key, NDArray> pull_buf_;
  std::unordered_map<ps::Key, NDArray> pull_prev_buf_;
  bool partial_pull_;
  bool partial_pull_history_;
  double partial_pull_history_alpha_;
};

// for UDP server
class KVCheapStoreDist : public KVStoreLocal {
public:
 explicit KVCheapStoreDist(bool use_device_comm)
     : KVStoreLocal(use_device_comm), ps_worker_(nullptr), server_(nullptr) {
  //  //debug
  //  LG << "Using KVCheapStoreDist!";

   if (IsWorkerNode()) {
     ps_worker_ = new ps::KVCheapWorker<real_t>(0);
     ps::StartAsync("mxnet\0");
     if (!ps::Postoffice::Get()->is_recovery()) {
       ps::Postoffice::Get()->Barrier(
         ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
     }
   }
   bigarray_bound_ = dmlc::GetEnv("MXNET_KVSTORE_BIGARRAY_BOUND", 1000 * 1000);

   // // debug
   // LG << "bigarray_bound_=" << bigarray_bound_;
 }

 virtual ~KVCheapStoreDist() {
   Engine::Get()->WaitForAll();
   if (IsWorkerNode()) {
     if (barrier_before_exit_) {
       Barrier();
       if (get_rank() == 0) {
         // stop the executor at servers
         SendCommandToServers(kStopServer, "");
       }
     }
     ps::Finalize(barrier_before_exit_);
     delete ps_worker_;
   }
 }

 void Init(const std::vector<int>& keys,
           const std::vector<NDArray>& values) override {
   CheckUnique(keys);
   for (size_t i = 0; i < keys.size(); ++i) {
     comm_->Init(keys[i], values[i].shape(), values[i].dtype());
   }
   if (get_rank() == 0) {
     Push_(keys, values, 0, false);
     // wait until the push is finished
     for (const auto& v : values) {
       v.WaitToWrite();
     }
   } else {
     // do nothing
   }
   if (!ps::Postoffice::Get()->is_recovery()) {
     Barrier();
   }
 }

 void Push(const std::vector<int>& keys,
           const std::vector<NDArray>& values,
           int priority) override {
   Push_(keys, values, priority, true);
 }

 void Pull(const std::vector<int>& keys,
           const std::vector<NDArray*>& values,
           int priority) override {
   std::vector<int> uniq_keys;
   std::vector<std::vector<NDArray*> > grouped_vals;
   GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

   for (size_t i = 0; i < uniq_keys.size(); ++i) {
     int key = uniq_keys[i];
     // use the same array for merging to guarantee that pull always happens
     // after the previous push on this key
     auto& recv_buf = comm_buf_[key];
     if (recv_buf.is_none()) {
       // it may happen for the first time a no-rank-0 worker pull the weight.
       recv_buf = NDArray(
         grouped_vals[i][0]->shape(), pinned_ctx_, true, grouped_vals[i][0]->dtype());
     }
     auto pull_from_servers = [this, key, recv_buf](
         RunContext rctx, Engine::CallbackOnComplete cb) {
       // convert to ps keys
       size_t size = recv_buf.shape().Size();
       // LG << "EncodeKey in Pull";
       PSKV& pskv = EncodeKey(key, size);
#if MKL_EXPERIMENTAL == 1
       mkl_set_tblob_eager_mode(recv_buf.data());
#endif
       real_t* data = static_cast<real_t*>(recv_buf.data().dptr_);
       // false means not to delete data when SArray is deleted
       auto vals = new ps::SArray<real_t>(data, size, false);
       // issue pull
       CHECK_NOTNULL(ps_worker_)->ZPull(
           pskv.keys, vals, &pskv.lens, 0, [vals, cb](){ delete vals; cb(); });
     };

     CHECK_NOTNULL(Engine::Get())->PushAsync(
         pull_from_servers,
         pinned_ctx_,
         {},
         {recv_buf.var()},
         FnProperty::kNormal,
         priority,
         PROFILER_MESSAGE("KVStoreDistPull"));

     comm_->Broadcast(key, recv_buf, grouped_vals[i], priority);
   }
 }

 void set_updater(const Updater& updater) override {
   CHECK(updater) << "invalid updater";
   if (IsServerNode()) {
     CHECK_NOTNULL(server_)->set_updater(updater);
   } else {
     updater_ = updater;
   }
 }

 void Barrier() override {
   ps::Postoffice::Get()->Barrier(ps::kWorkerGroup);
 }


 void SendCommandToServers(int cmd_id,
                           const std::string& cmd_body) override {
   CHECK_NOTNULL(ps_worker_);
   ps_worker_->Wait(ps_worker_->Request(cmd_id, cmd_body, ps::kServerGroup));
 }

 int get_group_size() const override { return ps::NumWorkers(); }

 int get_rank() const override { return ps::MyRank(); }

 int get_num_dead_node(int node_id, int timeout) const override {
   int number = 0;
   auto dead_nodes = ps::Postoffice::Get()->GetDeadNodes(timeout);
   const auto& watch_nodes = ps::Postoffice::Get()->GetNodeIDs(node_id);
   std::unordered_set<int> watch_set(watch_nodes.begin(), watch_nodes.end());
   for (int r : dead_nodes) {
     if (watch_set.find(r) != watch_set.end()) number++;
   }
   return number;
 }

 void RunServer(const Controller& controller) override {
   CHECK(!IsWorkerNode());
   if (IsServerNode()) {
     server_ = new KVStoreDistServer();
     server_->set_controller(controller);
   }

   ps::StartAsync("mxnet_server\0");
   if (!ps::Postoffice::Get()->is_recovery()) {
     ps::Postoffice::Get()->Barrier(
       ps::kWorkerGroup + ps::kServerGroup + ps::kScheduler);
   }
   if (server_) server_->Run();
   ps::Finalize();
   if (server_) {
     delete server_;
   }
   server_ = nullptr;
 }

private:
 void Push_(const std::vector<int>& keys,
            const std::vector<NDArray>& values,
            int priority,
            bool do_merge)  {
   // first aggregate the values over keys
   std::vector<int> uniq_keys;
   std::vector<std::vector<NDArray> > grouped_vals;
   GroupKVPairs(keys, values, &uniq_keys, &grouped_vals);

   for (size_t i = 0; i < uniq_keys.size(); ++i) {
     // merge over devcies
     int key = uniq_keys[i];
     const auto& vals = grouped_vals[i];
     NDArray merged = do_merge ? comm_->Reduce(key, vals, priority) : vals[0];

     auto& send_buf = comm_buf_[key];
     if (merged.ctx().dev_mask() == cpu::kDevMask) {
       send_buf = merged;  // avoid memory copy
     } else {
       if (send_buf.is_none()) {
         send_buf = NDArray(merged.shape(), pinned_ctx_, true, merged.dtype());
       }
       CopyFromTo(merged, &send_buf);
     }

     // push to servers
     auto push_to_servers =
         [this, key, send_buf](RunContext rctx, Engine::CallbackOnComplete cb) {
       // convert to ps keys
       size_t size = send_buf.shape().Size();

       // debug
       // LG << "EncodeKey in Push_";
       PSKV& pskv = EncodeKey(key, size);

#if MKL_EXPERIMENTAL == 1
       mkl_set_tblob_eager_mode(send_buf.data());
#endif
       real_t* data = static_cast<real_t*>(send_buf.data().dptr_);
       // do push. false means no delete
       ps::SArray<real_t> vals(data, size, false);
      //  // debug
      //  LG << "ZPush called";
       CHECK_NOTNULL(ps_worker_)->ZPush(
           pskv.keys, vals, pskv.lens, 0, [cb]() { cb(); });
     };
     Engine::Get()->PushAsync(
         push_to_servers,
         pinned_ctx_,
         {send_buf.var()},
         {},
         FnProperty::kNormal,
         priority,
         PROFILER_MESSAGE("KVStoreDistPush"));
   }
 }

 /**
  * \brief check if the keys are all unique
  */
 void CheckUnique(const std::vector<int>& keys) {
   auto keys_copy = keys;
   auto last = std::unique(keys_copy.begin(), keys_copy.end());
   CHECK_EQ(static_cast<size_t>(std::distance(keys_copy.begin(), last)),
            static_cast<size_t>(keys.size()));
 }

 /**
  * \brief struct for ps keys and lens
  */
  struct PSKV {
    ps::SArray<ps::Key> keys;  // n keys
    ps::SArray<int> lens;  // the length of the i-th value
    int size;
  };

 /**
  * \brief cache all key partitions
  */
 std::unordered_map<int, PSKV> ps_kv_;

 /**
  * \brief serizelize EncodeKey
  */
 std::mutex mu_;

 /**
  * \brief convert to keys in ps
  */
 inline PSKV& EncodeKey(int key, size_t size) {
   mu_.lock();
   PSKV& pskv = ps_kv_[key];
   mu_.unlock();

   if (!pskv.keys.empty()) {
     CHECK_EQ(static_cast<size_t>(pskv.size), size) << "The value size cannot be changed";
   } else {
     auto krs = ps::Postoffice::Get()->GetServerKeyRanges();
     int num_keyranges = krs.size();
     CHECK_GT(num_keyranges, 0);

     // a simple heuristic for load balance
     if (size < bigarray_bound_) {
       // send it to a single random picked server
       int range_rank = (key * 9973) % num_keyranges;
       ps::Key ps_key = krs[range_rank].begin() + key;
       CHECK_LT(ps_key, krs[range_rank].end());
       pskv.keys.push_back(ps_key);
       pskv.lens.push_back(size);
       pskv.size = size;
     } else {
       // parition it to all servers
       pskv.size = 0;
       // make the chunck size <= bigarray_bound_
       size_t num_chunks = static_cast<size_t>(ceil(static_cast<double>(size)/bigarray_bound_));

       CHECK_LE(num_chunks, num_keyranges);

       for (int i = 0; i < num_chunks; ++i) {
         size_t part_size =
             static_cast<size_t>(round(static_cast<double>(size)/num_chunks*(i+1))) -
             static_cast<size_t>(round(static_cast<double>(size)/num_chunks*i));
         ps::Key ps_key = krs[i].begin() + key;  
         CHECK_LT(ps_key, krs[i].end());   
         pskv.keys.push_back(ps_key);
         pskv.lens.push_back(part_size);
         pskv.size += part_size;
       }

       CHECK_EQ(static_cast<size_t>(pskv.size), size);
     }
   }
   return pskv;
 }

 /**
  * \brief for worker to push and pull data
  */
  ps::KVCheapWorker<real_t>* ps_worker_;
 /**
  * \brief the server handle
  */
 KVStoreDistServer* server_;
 /**
  * \brief threshold for partition
  */
 size_t bigarray_bound_;
 /// \brief send & recver buffer
 std::unordered_map<int, NDArray> comm_buf_;
};

}  // namespace kvstore
}  // namespace mxnet


#endif  // MXNET_KVSTORE_KVSTORE_DIST_H_
