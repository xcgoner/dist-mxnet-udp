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

/*!
 * \file mxnet_node.h
 * \brief implement mxnet nodes
 */
#ifndef MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#define MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
#include <queue>
#include <string>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>
#include <future>
#include <vector>
#include <iterator>
#include "ps/ps.h"
#include "mxnet/kvstore.h"

namespace mxnet {
namespace kvstore {

static const int kStopServer = -1;
static const int kSyncMode = -2;

/**
 * \brief executor runs a function using the thread called \ref Start
 */
class Executor {
 public:
  /**
   * \brief start the executor
   */
  void Start() {
    std::unique_lock<std::mutex> lk(mu_);
    while (true) {
      cond_.wait(lk, [this]{return !queue_.empty();});
      Block blk = std::move(queue_.front());
      queue_.pop();
      lk.unlock();

      if (blk.f) {
        blk.f(); blk.p->set_value();
      } else {
        blk.p->set_value(); break;
      }
      lk.lock();
    }
  }

  /**
   * \brief function
   */
  typedef std::function<void()> Func;

  /**
   * \brief let the thread called \ref Start to exec a function. threadsafe
   */
  void Exec(const Func& func) {
    Block blk(func);
    auto fut = blk.p->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      queue_.push(std::move(blk));
      cond_.notify_one();
    }
    fut.wait();
  }

  /**
   * \brief stop the thread, threadsafe
   */
  void Stop() {
    Exec(Func());
  }

 private:
  struct Block {
  explicit Block(const Func& func) : f(func), p(std::make_shared<std::promise<void>>()) { }
    Func f;
    std::shared_ptr<std::promise<void>> p;
  };
  std::queue<Block> queue_;
  std::mutex mu_;
  std::condition_variable cond_;
};

class KVStoreDistServer {
 public:
  KVStoreDistServer() {
    using namespace std::placeholders;
    ps_server_ = new ps::KVServer<float>(0);
    static_cast<ps::SimpleApp*>(ps_server_)->set_request_handle(
        std::bind(&KVStoreDistServer::CommandHandle, this, _1, _2));
    ps_server_->set_request_handle(
        std::bind(&KVStoreDistServer::DataHandle, this, _1, _2, _3));
    sync_mode_ = false;
    merge_threshold_ = dmlc::GetEnv("MXNET_MERGE_THRESHOLD", (size_t)ps::NumWorkers());
    if (merge_threshold_ > ps::NumWorkers()) merge_threshold_ = ps::NumWorkers();
    tau_millisec_ = dmlc::GetEnv("MXNET_MERGE_TAU_MILLISECOND", 0);
    // debug
    LG << "merge_threshold_ = " << merge_threshold_;
    LG << "tau_millisec_ = " << tau_millisec_;
  }

  ~KVStoreDistServer() {
    delete ps_server_;
  }

  void set_controller(const KVStore::Controller& controller) {
    CHECK(controller);
    controller_ = controller;
  }

  void set_updater(const KVStore::Updater& updater)  {
    CHECK(updater);
    updater_ = updater;
  }

  /**
   * \brief blocked until received the command \a kSyncMode
   */
  void Run() {
    exec_.Start();
  }

 private:
  void CommandHandle(const ps::SimpleData& recved, ps::SimpleApp* app) {
    if (recved.head == kStopServer) {
      exec_.Stop();
    } else if (recved.head == kSyncMode) {
      sync_mode_ = true;
    } else {
      // let the main thread to execute ctrl, which is necessary for python
      exec_.Exec([this, recved]() {
          CHECK(controller_);
          controller_(recved.head, recved.body);
        });
    }
    app->Response(recved);
  }

  void DataHandle(const ps::KVMeta& req_meta,
                  const ps::KVPairs<real_t>& req_data,
                  ps::KVServer<real_t>* server) {

    // //debug 
    // LG << "keys: ";
    // for (int i = 0; i < req_data.keys.size(); ++i) {
    //   LG << req_data.keys[i];
    // }

    // // debug
    // std::ostringstream key_list;
    // key_list << "key_list: ";
    // for (int i = 0; i < req_data.keys.size(); ++i) {
    //   key_list << req_data.keys[i] << ", ";
    // }
    // LG << key_list.str();

    // do some check
    CHECK_EQ(req_data.keys.size(), (size_t)1);
    if (req_meta.push && !req_meta.fake) {
      CHECK_EQ(req_data.lens.size(), (size_t)1);
      CHECK_EQ(req_data.vals.size(), (size_t)req_data.lens[0]);
    }

    // int key = DecodeKey(req_data.keys[0]);
    ps::Key key = req_data.keys[0];
    auto& stored = store_[key];

    // initialize?
    if (store_iteration_.count(key) == 0) store_iteration_[key] = 0;

    // // debug
    // if (ps::IsServer())
    //   LOG(INFO) << "Server\t"<< ps::MyRank() << "\tKey:\t" << key << "\tValue:\t" << stored.shape();

    // there used several WaitToRead, this is because \a recved's memory
    // could be deallocated when this function returns. so we need to make sure
    // the operators with \a NDArray are actually finished
    if (req_meta.push) {
      if (req_meta.fake) {
        // must be in sync mode
        if (req_data.iteration == store_iteration_[key]) {
          // // debug
          // std::ostringstream sender_list;
          // sender_list << "sender_list: ";
          // for (auto sender : sender_list_[key]) {
          //   sender_list << sender << ", ";
          // }
          // LG << sender_list.str();
          // sender_list_[key].clear();

          // deubg
          // if (merge_num_[key] > merge_threshold_) LG << "timeout merged: " << merge_num_[key];
          // LG << "timeout merged: " << merge_num_[key];

          auto& merged = merge_buf_[key];

          int num_workers = ps::NumWorkers();
          int merge_num = merge_num_[key];
          CHECK_LE(merge_num, num_workers);
          CHECK_GE(merge_num, 1);
          if (merge_num != num_workers) {
            merged.array *= ( ((double)num_workers) / merge_num );
          }

          if (updater_) {
            exec_.Exec([this, key, &merged, &stored](){
                CHECK(updater_);
                updater_(key, merged.array, &stored);
              });
          } else {
            // if no updater, just copy
            CopyFromTo(merged.array, &stored);
          }
          for (const auto& req : merged.request) {
            server->Response(req);
          }
          merged.request.clear();
          merge_num_[key] = 0;
          store_iteration_[key] = store_iteration_[key] + 1;
          stored.WaitToRead();

          // deubg
          // LG << "timeout merged: " << merge_num_[key];
        }
        // else {
        //   // debug
        //   LG << "timeout ignored";
        // }
      }
      else {
        // message is not fake
        size_t ds[] = {(size_t)req_data.lens[0]};
        TShape dshape(ds, ds + 1);
        TBlob recv_blob((real_t*)req_data.vals.data(), // NOLINT(*)
                        dshape, cpu::kDevMask);
        NDArray recved = NDArray(recv_blob, 0);
        // LG << "push request received: key = " << key;
        if (stored.is_none()) {
          // the very first push
          // initialization
          stored = NDArray(dshape, Context());
          CopyFromTo(recved, &stored, 0);
          // initialize the iteration counter
          store_iteration_[key] = 0;
          server->Response(req_meta);
          stored.WaitToRead();
        } else if (sync_mode_) {
          // synced push
  
          // TODO: check req_data.iteration
          // LG << key << " push itr: " << req_data.iteration << ", current itr: " << store_iteration_[key];
  
          if (req_data.iteration == store_iteration_[key]) {
            // debug
            sender_list_[key].push_back(req_meta.sender);
  
            auto& merged = merge_buf_[key];
            if (merged.array.is_none()) {
              merged.array = NDArray(dshape, Context());
            }
    
            if (merged.request.size() == 0) {
              CopyFromTo(recved, &merged.array, 0);
              merge_num_[key] = 1;
            } else {
              merged.array += recved;
              merge_num_[key] = merge_num_[key] + 1;
            }
    
            merged.request.push_back(req_meta);
    
            // if (merged.request.size() == (size_t)ps::NumWorkers()) {
            if (merged.request.size() >= (size_t)merge_threshold_) {
              // // debug
              // LG << "merging";
              // let the main thread to execute updater_, which is necessary for
              // python

              if (tau_millisec_ > 0 && merged.request.size() != (size_t)ps::NumWorkers()) {
                // start the timer
                // TODO: check tau_millisec positive
                std::thread timer([](const ps::KVMeta& req_meta, const ps::KVPairs<real_t>& req_data, int tau_millisec) {
                  // set timeout
                  std::this_thread::sleep_for(std::chrono::milliseconds(tau_millisec));
                  // send fake message
                  ps::Message msg;
                  msg.meta.head        = req_meta.cmd;
                  msg.meta.push        = true;
                  msg.meta.sender      = req_meta.sender;
                  msg.meta.timestamp   = req_meta.timestamp;
                  msg.meta.iteration   = req_data.iteration;
                  msg.meta.customer_id = req_meta.customer_id;
                  msg.meta.fake        = true;
                  msg.meta.request     = true;
    
                  msg.AddData(req_data.keys);
    
                  auto* obj = ps::Postoffice::Get()->GetCustomer(req_meta.customer_id, 5);
                  CHECK(obj) << "timeout (5 sec) to wait App " << req_meta.customer_id << " ready";
                  // insert to the head of the queue
                  // the queue is thread-safe
                  obj->AcceptPriority(msg);
                }, req_meta, req_data, tau_millisec_);
                timer.detach();
                merged.array.WaitToRead();
              }
              else {
                // // debug
                // std::ostringstream sender_list;
                // sender_list << "sender_list: ";
                // for (auto sender : sender_list_[key]) {
                //   sender_list << sender << ", ";
                // }
                // LG << sender_list.str();
                // sender_list_[key].clear();

                // deubg
                // LG << "normally merged!";
    
                int num_workers = ps::NumWorkers();
                if (merge_threshold_ != num_workers) {
                  merged.array *= ( ((double)num_workers) / merge_threshold_ );
                }
    
                if (updater_) {
                  exec_.Exec([this, key, &merged, &stored](){
                      CHECK(updater_);
                      updater_(key, merged.array, &stored);
                    });
                } else {
                  // if no updater, just copy
                  CopyFromTo(merged.array, &stored);
                }
                for (const auto& req : merged.request) {
                  server->Response(req);
                }
                merged.request.clear();
                merge_num_[key] = 0;
                store_iteration_[key] = store_iteration_[key] + 1;
                stored.WaitToRead();

                // // deubg
                // LG << "normally merged!";
                // LG << "number of keys:" << merge_num_.size();
              }
            } else {
              merged.array.WaitToRead();
            }
          }
          else {
            // ignore the delayed updates
            server->Response(req_meta);
            // debug 
            // LG << "push is ignored";
          }
        } else {
          // async push
          exec_.Exec([this, key, &recved, &stored](){
              CHECK(updater_);
              updater_(key, recved, &stored);
            });
          server->Response(req_meta);
          stored.WaitToRead();
        }
      }
    } else {
      // pull
      if (sync_mode_ && req_data.iteration == store_iteration_[key] + 1) {
        LG << key << " pull itr: " << req_data.iteration << ", current itr: " << store_iteration_[key] << ", push back";
        // push back to the queue and process later
        ps::Message msg;
        msg.meta.head        = req_meta.cmd;
        msg.meta.push        = false;
        msg.meta.sender      = req_meta.sender;
        msg.meta.timestamp   = req_meta.timestamp;
        msg.meta.iteration   = req_data.iteration;
        msg.meta.customer_id = req_meta.customer_id;
        msg.meta.fake        = false;
        msg.meta.request     = true;

        msg.AddData(req_data.keys);
        msg.AddData(req_data.vals);
        auto* obj = ps::Postoffice::Get()->GetCustomer(req_meta.customer_id, 5);
        CHECK(obj) << "timeout (5 sec) to wait App " << req_meta.customer_id << " ready";
        // insert to the tail of the queue
        // the queue is thread-safe
        obj->Accept(msg);
      }
      else if( (sync_mode_ && req_data.iteration == store_iteration_[key]) || !sync_mode_) {
        // LG << key << " pull itr: " << req_data.iteration << ", current itr: " << store_iteration_[key] << ", response";
        // LG << "pull request received: key = " << key;
        ps::KVPairs<real_t> response;
        CHECK(!stored.is_none()) << "init " << key << " first";
        int len = stored.shape()[0];
        response.keys = req_data.keys;
        response.lens = {len};
        // TODO(mli) try to remove this CopyFrom
        response.vals.CopyFrom(static_cast<const float*>(stored.data().dptr_), len);
        response.iteration = store_iteration_[key];
        // // debug
        // LG << "store_iteration_[key]: " << store_iteration_[key];
        server->Response(req_meta, response);
        // debug
        // LG << "pull response sent: key = " << key;
      }
      else {
        // TODO: ???
        LG << "something is wrong for pulling" << "req_data.iteration: " << req_data.iteration << ", store_iteration_[key]: " << store_iteration_[key];
        // still response
        ps::KVPairs<real_t> response;
        CHECK(!stored.is_none()) << "init " << key << " first";
        int len = stored.shape()[0];
        response.keys = req_data.keys;
        response.lens = {len};
        // TODO(mli) try to remove this CopyFrom
        response.vals.CopyFrom(static_cast<const float*>(stored.data().dptr_), len);
        response.iteration = store_iteration_[key];
        // // debug
        // LG << "store_iteration_[key]: " << store_iteration_[key];
        server->Response(req_meta, response);
      }
    }
  }

  int DecodeKey(ps::Key key) {
    auto kr = ps::Postoffice::Get()->GetServerKeyRanges()[ps::MyRank()];
    return key - kr.begin();
  }

  /**
   * \brief user defined
   */
  bool sync_mode_;
  KVStore::Controller controller_;
  KVStore::Updater updater_;

  // std::unordered_map<int, NDArray> store_;
  std::unordered_map<ps::Key, NDArray> store_;
  std::unordered_map<ps::Key, int> store_iteration_;

  struct MergeBuf {
    std::vector<ps::KVMeta> request;
    NDArray array;
  };
  // std::unordered_map<int, MergeBuf> merge_buf_;
  std::unordered_map<ps::Key, MergeBuf> merge_buf_;

  std::unordered_map<ps::Key, int> merge_num_;
  // debug
  std::unordered_map<ps::Key, std::vector<int>> sender_list_;

  int merge_threshold_;

  Executor exec_;

  ps::KVServer<float>* ps_server_;

  // timer
  // threshold
  // // second
  // std::chrono::seconds tau_sec_;
  // // nanosecond
  // std::chrono::nanoseconds tau_nsec_;
  // millisecond
  int tau_millisec_;
};

}  // namespace kvstore
}  // namespace mxnet

#endif  // MXNET_KVSTORE_KVSTORE_DIST_SERVER_H_
