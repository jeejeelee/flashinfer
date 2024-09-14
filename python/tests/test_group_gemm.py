"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import flashinfer
import pytest
import torch


DTYPES = [torch.bfloat16]
CUDA_DEVICES = ["cuda:0"]


@pytest.mark.parametrize("batch_size", [1, 77, 199])
@pytest.mark.parametrize("num_rows_per_batch", [3, 10, 99])
@pytest.mark.parametrize("d_in", [128, 1024, 4096])
@pytest.mark.parametrize("d_out", [128, 1024, 4096])
@pytest.mark.parametrize("use_weight_indices", [False, True])
@pytest.mark.parametrize("column_major", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", CUDA_DEVICES)
def test_segment_gemm(
    batch_size,
    num_rows_per_batch,
    d_in,
    d_out,
    use_weight_indices,
    column_major,
    dtype,
    device,
    # index,
):
    try:
        if batch_size * num_rows_per_batch > 8192:
            pytest.skip("batch_size * num_rows_per_batch too large for test.")
            # print("batch_size * num_rows_per_batch too large for test.")
            # return
        index = 0
        torch.manual_seed(42)
        workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8).to(device)
        segment_gemm = flashinfer.gemm.SegmentGEMMWrapper(workspace_buffer)
        x = torch.randn(batch_size * num_rows_per_batch, d_in, dtype=dtype).to(device)
        if use_weight_indices:
            num_weights = 1024
            if column_major:
                weight = torch.randn(num_weights, d_out, d_in, dtype=dtype).to(device)
            else:
                weight = torch.randn(num_weights, d_in, d_out, dtype=dtype).to(device)
        else:
            if column_major:
                weight = torch.randn(batch_size, d_out, d_in, dtype=dtype).to(device)
            else:
                weight = torch.randn(batch_size, d_in, d_out, dtype=dtype).to(device)
        # torch.cuda.nvtx.range_push(f"stage=4_no_{index}")
        y = segment_gemm.run(
            x,
            weight,
            batch_size,
            weight_column_major=column_major,
            seg_lens=torch.full((batch_size,), num_rows_per_batch, dtype=torch.int64),
            weight_indices=(
                (torch.arange(0, batch_size) % num_weights).to(device)
                if use_weight_indices
                else None
            ),
        )
        # torch.cuda.nvtx.range_pop()
        if use_weight_indices:
            for i in range(batch_size):
                torch.testing.assert_close(
                    y[i * num_rows_per_batch : (i + 1) * num_rows_per_batch],
                    torch.matmul(
                        x[i * num_rows_per_batch : (i + 1) * num_rows_per_batch],
                        (
                            weight[i % num_weights].T
                            if column_major
                            else weight[i % num_weights]
                        ),
                    ),
                    rtol=1e-3,
                    atol=1e-3,
                    msg="assertion failed at batch {}".format(i),
                )
        else:
            torch.testing.assert_close(
                y,
                torch.matmul(
                    x.view(batch_size, num_rows_per_batch, d_in),
                    weight.transpose(-1, -2) if column_major else weight,
                ).view(batch_size * num_rows_per_batch, d_out),
                rtol=1e-3,
                atol=1e-3,
            )
    except Exception as e:
        assert isinstance(e, torch.OutOfMemoryError)


if __name__ == "__main__":
    #     for i in range(10):
    #         test_segment_gemm(1, 99, 128, 1024, False, False, torch.bf16, "cuda:0", i)

    #     # test_segment_gemm(99, 99, 128, 1024, False, True, torch.float16, "cuda:0")
    #     # test_segment_gemm(32, 99, 128, 1024, True, False, torch.float16, "cuda:0")
    # #     # test_segment_gemm(16, 99, 128, 1024, True, True, torch.float16, "cuda:0")
    test_segment_gemm(
        batch_size=1,
        num_rows_per_batch=99,
        d_in=4096,
        d_out=1024,
        use_weight_indices=False,
        column_major=False,
        dtype=torch.bfloat16,
        device="cuda:0",
    )
