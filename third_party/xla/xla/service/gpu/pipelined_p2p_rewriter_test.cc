/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/service/gpu/pipelined_p2p_rewriter.h"

#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using ::testing::HasSubstr;

class PipelinedP2pRewriterTest : public HloTestBase {
 private:
  int64_t GetIndex(
      absl::string_view hlo_name,
      const std::vector<HloInstruction*>& instruction_sequence) const {
    return absl::c_find_if(instruction_sequence,
                           [hlo_name](HloInstruction* instruction) {
                             return instruction->name() == hlo_name;
                           }) -
           instruction_sequence.begin();
  }

 protected:
  void VerifyBefore(const std::vector<HloInstruction*>& instructions,
                    absl::string_view a, absl::string_view b) {
    EXPECT_LT(GetIndex(a, instructions), GetIndex(b, instructions));
  }
  void VerifySubstring(HloModule* module, absl::string_view instr_name,
                       absl::string_view substring) {
    HloInstruction* instr = FindInstruction(module, instr_name);
    EXPECT_THAT(instr->ToString(), HasSubstr(substring));
  }
};

// Tests the rewrite for a pipelined Send/Recv chain with only one channel
// group.
TEST_F(PipelinedP2pRewriterTest, LHSSendRecvPipelined1) {
  const char* kModuleStr = R"(
  HloModule test, is_scheduled=true

  while-cond {
    param = (u32[], (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(25)
    ROOT cond-result = pred[] compare(count, ub), direction=LT
  }

  while-body {
    param = (u32[], (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0

    recv-done.q = (f32[1,1024,1024], token[]) get-tuple-element(param), index=1
    recv-data = f32[1, 1024, 1024] get-tuple-element(recv-done.q), index=0

    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)
    replica = u32[] replica-id()
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    p = f32[1, 1024, 1024] broadcast(conv), dimensions={}
    b = f32[1, 1024, 1024] add(p, recv-data)
    c = f32[1, 1024, 1024] multiply(b, b)
    d = f32[1, 1024, 1024] tan(c)
    s = f32[1, 1024, 1024] dot(c, d), lhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    send-data = f32[1, 1024, 1024] add(c, s)

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    send = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.p = (f32[1,1024,1024], token[]) recv-done(recv), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.p = token[] send-done(send), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    ROOT body-result = (u32[], (f32[1,1024,1024], token[]), token[])
      tuple(new-count, recv-done.p, send-done.p)
  }

  ENTRY main {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all.1 = token[] after-all()
    recv.1 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    send.1 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}, {3,4}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.1.p = (f32[1,1024,1024], token[]) recv-done(recv.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.1.p = token[] send-done(send.1), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    while-init.p =  (u32[], (f32[1,1024,1024], token[]), token[])
      tuple(c0, recv-done.1.p, send-done.1.p)
    while-result.p = (u32[], (f32[1,1024,1024], token[]), token[])
      while(while-init.p),
      body=while-body, condition=while-cond,
      backend_config={"known_trip_count":{"n":"25"}}

    recv-done.1.q = (f32[1,1024,1024], token[]) get-tuple-element(while-result.p), index=1

    ROOT entry-result = f32[1, 1024, 1024] get-tuple-element(recv-done.1.q), index=0
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  PipelinedP2PRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  SequentialHloOrdering order(module->schedule());
  const std::vector<HloInstruction*>& while_body =
      order.SequentialOrder(*module->GetComputationWithName("while-body"))
          ->instructions();
  const std::vector<HloInstruction*>& main =
      order.SequentialOrder(*module->GetComputationWithName("main"))
          ->instructions();

  // Verify the pipelined Send-Recv chain in main.
  VerifyBefore(main, "recv.1", "send.1");
  VerifyBefore(main, "send.1", "while-result");
  VerifyBefore(main, "while-result", "recv-done.1");
  VerifyBefore(main, "recv-done.1", "send-done.1");

  // Verify the pipelined Send-Recv chain in while-body.
  VerifyBefore(while_body, "recv-done", "send-done");
  VerifyBefore(while_body, "send-done", "recv");
  VerifyBefore(while_body, "recv", "send");

  // Verify that meta info in instructions are preserved.
  VerifySubstring(module.get(), "while-result",
                  "backend_config={\"known_trip_count\":{\"n\":\"25\"}}");

  auto verify_pipeline_attr = [&](const absl::string_view hlo_name) {
    VerifySubstring(module.get(), hlo_name,
                    "frontend_attributes={_xla_send_recv_pipeline=\"0\"}");
  };
  verify_pipeline_attr("send-done");
  verify_pipeline_attr("recv-done");
  verify_pipeline_attr("send-done.1");
  verify_pipeline_attr("recv-done.1");
}

// Tests the rewrite for a pipelined Send/Recv chain with two channel groups.
TEST_F(PipelinedP2pRewriterTest, LHSSendRecvPipelined2) {
  const char* kModuleStr = R"(
  HloModule test, is_scheduled=true

  while-cond {
    param = (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0
    ub = u32[] constant(25)
    ROOT cond-result = pred[] compare(count, ub), direction=LT
  }

  while-body {
    param = (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[]) parameter(0)
    count = get-tuple-element(param), index=0

    recv-done.0.q = (f32[1,1024,1024], token[]) get-tuple-element(param), index=1
    recv-data.0 = f32[1, 1024, 1024] get-tuple-element(recv-done.0.q), index=0
    recv-done.1.q = (f32[1,1024,1024], token[]) get-tuple-element(param), index=3
    recv-data.1 = f32[1, 1024, 1024] get-tuple-element(recv-done.1.q), index=0

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[1, 1024, 1024] broadcast(compare0), dimensions={}
    recv-data = f32[1, 1024, 1024] select(compare, recv-data.0, recv-data.1)

    c1 = u32[] constant(1)
    new-count = u32[] add(count, c1)
    c10 = u32[] constant(10)
    sum = u32[] add(replica, c10)
    sum2 = u32[] add(sum, count)
    conv = f32[] convert(sum2)
    p = f32[1, 1024, 1024] broadcast(conv), dimensions={}
    b = f32[1, 1024, 1024] add(p, recv-data)
    c = f32[1, 1024, 1024] multiply(b, b)
    d = f32[1, 1024, 1024] tan(c)
    s = f32[1, 1024, 1024] dot(c, d), lhs_batch_dims={0},
      lhs_contracting_dims={1}, rhs_batch_dims={0}, rhs_contracting_dims={1}
    send-data = f32[1, 1024, 1024] add(c, s)

    after-all = token[] after-all()
    recv = (f32[1, 1024, 1024], u32[], token[]) recv(after-all), channel_id=1,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    send = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all),
      channel_id=1, frontend_attributes={
        _xla_send_recv_source_target_pairs="{{3,0}}",
        _xla_send_recv_pipeline="0"
      }
    recv-done.p = (f32[1,1024,1024], token[]) recv-done(recv), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.p = token[] send-done(send), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }

    after-all.1 = token[] after-all()
    recv.1 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
        _xla_send_recv_pipeline="1"
      }
    send.1 = (f32[1, 1024, 1024], u32[], token[]) send(send-data, after-all.1),
      channel_id=2, frontend_attributes={
       _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
       _xla_send_recv_pipeline="1"
      }
    recv-done.1.p = (f32[1,1024,1024], token[]) recv-done(recv.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }
    send-done.1.p = token[] send-done(send.1), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }

    ROOT body-result = (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[])
      tuple(new-count, recv-done.p, send-done.p, recv-done.1.p, send-done.1.p)
  }

  ENTRY main {
    c0 = u32[] constant(0)
    f0 = f32[] constant(0.0)
    init = f32[1, 1024, 1024] broadcast(f0), dimensions={}

    after-all.2 = token[] after-all()
    recv.2 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.2), channel_id=1,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{3,0}}",
       _xla_send_recv_pipeline="0"
    }
    send.2 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.2), channel_id=1,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{3,0}}",
       _xla_send_recv_pipeline="0"
    }
    recv-done.2.p = (f32[1,1024,1024], token[]) recv-done(recv.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }
    send-done.2.p = token[] send-done(send.2), channel_id=1,
      frontend_attributes={
        _xla_send_recv_pipeline="0"
      }

    after-all.3 = token[] after-all()
    recv.3 = (f32[1, 1024, 1024], u32[], token[]) recv(after-all.3), channel_id=2,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
       _xla_send_recv_pipeline="1"
    }
    send.3 = (f32[1, 1024, 1024], u32[], token[]) send(init, after-all.3), channel_id=2,
      frontend_attributes={
       _xla_send_recv_source_target_pairs="{{0,1}, {1,2}, {2,3}}",
       _xla_send_recv_pipeline="1"
    }
    recv-done.3.p = (f32[1,1024,1024], token[]) recv-done(recv.3), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }
    send-done.3.p = token[] send-done(send.3), channel_id=2,
      frontend_attributes={
        _xla_send_recv_pipeline="1"
      }

    while-init.p =  (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[]) tuple(c0, recv-done.2.p, send-done.2.p, recv-done.3.p, send-done.3.p)
    while-result.p = (u32[], (f32[1,1024,1024], token[]), token[],
      (f32[1,1024,1024], token[]), token[]) while(while-init.p),
      body=while-body, condition=while-cond,
      backend_config={"known_trip_count":{"n":"25"}}

    recv-done.2.q = (f32[1,1024,1024], token[]) get-tuple-element(while-result.p), index=1
    recv-data.2 = f32[1, 1024, 1024] get-tuple-element(recv-done.2.q), index=0
    recv-done.3.q = (f32[1,1024,1024], token[]) get-tuple-element(while-result.p), index=3
    recv-data.3 = f32[1, 1024, 1024] get-tuple-element(recv-done.3.q), index=0

    replica = u32[] replica-id()
    constant0 = u32[] constant(0)
    compare0 = pred[] compare(replica, constant0), direction=EQ
    compare = pred[1, 1024, 1024] broadcast(compare0), dimensions={}
    ROOT entry-result = f32[1, 1024, 1024] select(compare, recv-data.2, recv-data.3)
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  PipelinedP2PRewriter rewriter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  SequentialHloOrdering order(module->schedule());
  const std::vector<HloInstruction*>& while_body =
      order.SequentialOrder(*module->GetComputationWithName("while-body"))
          ->instructions();
  const std::vector<HloInstruction*>& main =
      order.SequentialOrder(*module->GetComputationWithName("main"))
          ->instructions();

  // Verify the pipelined Send-Recv chain in main.
  VerifyBefore(main, "recv.2", "send.2");
  VerifyBefore(main, "send.2", "recv.3");
  VerifyBefore(main, "recv.3", "send.3");
  VerifyBefore(main, "send.3", "while-result");
  VerifyBefore(main, "while-result", "recv-done.2");
  VerifyBefore(main, "recv-done.2", "recv-done.3");
  VerifyBefore(main, "recv-done.3", "send-done.2");
  VerifyBefore(main, "send-done.2", "send-done.3");

  // Verify the pipelined Send-Recv chain in while-body.
  VerifyBefore(while_body, "recv-done", "recv-done.1");
  VerifyBefore(while_body, "recv-done.1", "send-done");
  VerifyBefore(while_body, "send-done", "send-done.1");
  VerifyBefore(while_body, "send-done.1", "recv");
  VerifyBefore(while_body, "recv", "send");
  VerifyBefore(while_body, "send", "recv.1");
  VerifyBefore(while_body, "recv.1", "send.1");

  // Verify that meta info in instructions are preserved.
  VerifySubstring(module.get(), "while-result",
                  "backend_config={\"known_trip_count\":{\"n\":\"25\"}}");

  auto verify_pipeline_attr = [&](const absl::string_view hlo_name,
                                  const std::string& pipeline) {
    VerifySubstring(
        module.get(), hlo_name,
        "frontend_attributes={_xla_send_recv_pipeline=\"" + pipeline + "\"}");
  };
  verify_pipeline_attr("send-done", "0");
  verify_pipeline_attr("recv-done", "0");
  verify_pipeline_attr("send-done.1", "1");
  verify_pipeline_attr("recv-done.1", "1");
  verify_pipeline_attr("send-done.2", "0");
  verify_pipeline_attr("recv-done.2", "0");
  verify_pipeline_attr("send-done.3", "1");
  verify_pipeline_attr("recv-done.3", "1");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
