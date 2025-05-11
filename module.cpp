#include <ATen/ATen.h>
#include <cmath>
#include <immintrin.h>
#include <iostream>
#include <sys/time.h>
#include <time.h>
#include <torch/extension.h>
#include <vector>

// Uncomment for ISPC
// #include "module_ispc.h"
// using namespace ispc;

// ------------------------------------ //c
// 	WARM-UP: ACCESSING TENSORS      //
// ------------------------------------ //

// Step #1: Understand Read/Write Accessors for a 2D Tensor
inline float twoDimRead(std::vector<float> &tensor, const int &x, const int &y,
                        const int &sizeX) {
  // Note that sizeX is the size of a Row, not the number of rows
  return tensor[x * (sizeX) + y];
}

inline void twoDimWrite(std::vector<float> &tensor, const int &x, const int &y,
                        const int &sizeX, const float &val) {
  tensor[x * (sizeX) + y] = val;
}

// Step #2: Implement Read/Write Accessors for a 4D Tensor
inline float fourDimRead(std::vector<float> &tensor, const int &x, const int &y,
                         const int &z, const int &b, const int &sizeX,
                         const int &sizeY, const int &sizeZ) {

  return tensor[x * sizeX * sizeY * sizeZ + y * sizeY * sizeZ + z * sizeZ + b];
}

inline void fourDimWrite(std::vector<float> &tensor, const int &x, const int &y,
                         const int &z, const int &b, const int &sizeX,
                         const int &sizeY, const int &sizeZ, const float &val) {
 
  tensor[x * sizeX * sizeY * sizeZ + y * sizeY * sizeZ + z * sizeZ + b] = val;
}

// DO NOT EDIT THIS FUNCTION //
std::vector<float> formatTensor(torch::Tensor tensor) {
  tensor = tensor.flatten();
  tensor = tensor.contiguous();
  std::vector<float> vec(tensor.data_ptr<float>(),
                         tensor.data_ptr<float>() + tensor.numel());
  return vec;
}

/* Programming Your Attention Modules.
 *
 * You are given Q, K, and V Tensors as inputs that are formatted as vectors. We
 * have also created O and QK^t Tensors that are formatted as vectors. After you
 * have implemented your accessors in the Warm-Up you should be able to
 * read/write to these tensors via the read/write functions above.
 *
 * You are also given 4 integers as parameters: B, H, N, d:
 *
 * B (Batch Size) - The number of samples for your attention layer. Think of it
 * this way - if I asked my dnn a question and it output 5 different answers it
 * had a batch size of 5. These samples are independent of each other and thus
 * can be parallelized.
 *
 * H (Number of Heads) - Each head runs on its own set of Q, K, V matrices. This
 * effectively allows each head to operate the same attention algorithm, but
 * each with each head using different hyperparameters. These allow each head to
 * have their own definition of what relevance is when looking at a token. These
 * heads can operate independently of one another and thus can be parallized.
 *
 * N (Sequence Length) - The number of tokens. You may think of this as the
 * number of words in a sample.
 *
 * d (Embedding Dimensionality) - The number of features each token encodes per
 * attention head. Let's say I encoded a word using the follow (length, number
 * of vowels, has a capital letters). The emvedded dimensionaliy would be 3.
 * */

// ---------------------------------------------------------- //
//                  PART 1: NAIVE ATTENTION                   //
// ---------------------------------------------------------- //

torch::Tensor myNaiveAttention(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor QK_tTensor,
                               int B, int H, int N, int d) {
  // Q, K, V Shape: (B, H, N, d)
  // QK^t Shape: (N, N)
  // O Shape: (B, H, N, d)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

  // Format tensors into vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);
  std::vector<float> QK_t = formatTensor(QK_tTensor);

  // Temporary buffer for one row of QK^t and softmax
  std::vector<float> temp_row(N);

  // Use for loops for batch and heads
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      // Computing Q * K^T and softmax row-wise
      for (int i = 0; i < N; i++) {
        // Computing Q * K^T for row i
        float exp_sum = 0.0;
        int j = 0;
        while (j < N) { // While loop for column index
          float sum = 0.0;
          int k = 0;
          while (k < d) { // While loop for embedding dimension
            sum += fourDimRead(Q, b, h, i, k, H, N, d) *
                   fourDimRead(K, b, h, j, k, H, N, d);
            k++;
          }
          temp_row[j] = std::exp(sum);
          exp_sum += temp_row[j];
          j++;
        }

        // Apply softmax and store in QK_t
        j = 0;
        while (j < N) { // While loop for column index
          float softmax_val = temp_row[j] / exp_sum;
          twoDimWrite(QK_t, i, j, N, softmax_val);
          temp_row[j] = softmax_val; // Store for P * V
          j++;
        }
      }

      // Compute O = P * V
      for (int i = 0; i < N; i++) {
        int l = 0;
        while (l < d) { // While loop for embedding dimension
          float sum = 0.0;
          int k = 0;
          while (k < N) { // While loop for sequence length
            sum += twoDimRead(QK_t, i, k, N) *
                   fourDimRead(V, b, h, k, l, H, N, d);
            k++;
          }
          fourDimWrite(O, b, h, i, l, H, N, d, sum); // Fixed: Added d
          l++;
        }
      }
    }
  }

  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}


// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //

torch::Tensor myUnfusedAttentionBlocked(torch::Tensor QTensor,
                                        torch::Tensor KTensor,
                                        torch::Tensor VTensor,
                                        torch::Tensor QK_tTensor, int B, int H,
                                        int N, int d) {

  // Q, K, V are passed in with Shape: (B, H, N, d)
  // QK^t Intermediate Tensor has Shape (N, N)

  // Make O Tensor with Shape (B, H, N, d)
  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

  // Format O, Q, K, and V tensors into 4D vectors
  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);

  // Format QK_t Tensor into a 2D vector.
  std::vector<float> QK_t = formatTensor(QK_tTensor);

  // -------- YOUR CODE HERE  -------- //

  constexpr int L = 16;

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      // QK_t = Q * K^t
      for (int b_i = 0; b_i < N; b_i += L) {
        for (int b_j = 0; b_j < N; b_j += L) {
          for (int b_k = 0; b_k < d; b_k += L) {
            int m_i = std::min(N, b_i + L);
            int m_j = std::min(N, b_j + L);
            int m_k = std::min(d, b_k + L);
            for (int i = b_i; i < m_i; i++) {
              for (int j = b_j; j < m_j; j++) {
                float sum = twoDimRead(QK_t, i, j, N);
                for (int k = b_k; k < m_k; k++) {
                  sum += fourDimRead(Q, b, h, i, k, H, N, d) *
                         fourDimRead(K, b, h, j, k, H, N, d);
                }
                twoDimWrite(QK_t, i, j, N, sum);
              }
            }
          }
        }
      }

      // softmax(QK_t)
      for (int i = 0; i < N; i++) {
        float sum = 0.0;
        for (int j = 0; j < N; j++) {
          sum += std::exp(twoDimRead(QK_t, i, j, N));
        }
        for (int j = 0; j < N; j++) {
          float val = std::exp(twoDimRead(QK_t, i, j, N)) / sum;
          twoDimWrite(QK_t, i, j, N, val);
        }
      }

      // O = QK_t * V
      for (int b_i = 0; b_i < N; b_i += L) {
        for (int b_j = 0; b_j < d; b_j += L) {
          for (int b_k = 0; b_k < N; b_k += L) {
            int m_i = std::min(N, b_i + L);
            int m_j = std::min(d, b_j + L);
            int m_k = std::min(N, b_k + L);
            for (int i = b_i; i < m_i; i++) {
              for (int j = b_j; j < m_j; j++) {
                float sum = fourDimRead(O, b, h, i, j, H, N, d);
                for (int k = b_k; k < m_k; k++) {
                  sum += twoDimRead(QK_t, i, k, N) *
                         fourDimRead(V, b, h, k, j, H, N, d);
                }
                fourDimWrite(O, b, h, i, j, H, N, d, sum);
              }
            }
          }
        }
      }
    }
  }

  // DO NOT EDIT THIS RETURN STATEMENT //
  // It formats your C++ Vector O back into a Tensor of Shape (B, H, N, d) and
  // returns it //
  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}
// ---------------------------------------------------------- //
//     PART 2: BLOCKED MATRIX MULTIPLY AND UNFUSED SOFTMAX    //
// ---------------------------------------------------------- //


// ---------------------------------------------------------- //
//                 PART 3: FUSED ATTENTION                    //
// ---------------------------------------------------------- //

torch::Tensor myFusedAttention(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor temp, int B,
                               int H, int N, int d) {

  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);
  at::Tensor ORowTensor = at::zeros({N}, at::kFloat);

  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);
  std::vector<float> ORow = formatTensor(ORowTensor);

#pragma omp parallel for collapse(3)
  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      for (int i = 0; i < N; i++) {

        at::Tensor ORowTensor = temp.index({torch::indexing::Slice(
            omp_get_thread_num(), torch::indexing::None)});
        std::vector<float> ORow = formatTensor(ORowTensor);

        // QK_t = Q * K^t
        int j = 0;
        while (j < N) {
          float sum = 0.0;
          int k = 0;
          while (k < d) {
            sum += fourDimRead(Q, b, h, i, k, H, N, d) *
                   fourDimRead(K, b, h, j, k, H, N, d);
            k++;
          }
          ORow[j] = sum;
          j++;
        }

        // softmax(QK_t)
        float total = 0.0;
        j = 0;
        while (j < N) {
          ORow[j] = std::exp(ORow[j]);
          total += ORow[j];
          j++;
        }

        j = 0;
        while (j < N) {
          ORow[j] /= total;
          j++;
        }

        // O = softmax(QK_t) * V
        int jj = 0;
        while (jj < d) {
          float sum = 0.0;
          int kk = 0;
          while (kk < N) {
            sum += ORow[kk] * fourDimRead(V, b, h, kk, jj, H, N, d);
            kk++;
          }
          fourDimWrite(O, b, h, i, jj, H, N, d, sum);
          jj++;
        }
      }
    }
  }

  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}


// ---------------------------------------------------------- //
//                PART 4: FLASH ATTENTION 		      //
// ---------------------------------------------------------- //

torch::Tensor myFlashAttention(torch::Tensor QTensor, torch::Tensor KTensor,
                               torch::Tensor VTensor, torch::Tensor QiTensor,
                               torch::Tensor KjTensor, torch::Tensor VjTensor,
                               torch::Tensor SijTensor, torch::Tensor PijTensor,
                               torch::Tensor PVTensor, torch::Tensor OiTensor,
                               torch::Tensor LTensor, torch::Tensor LiTensor,
                               torch::Tensor LijTensor,
                               torch::Tensor LnewTensor, int Bc, int Br, int B,
                               int H, int N, int d) {

  at::Tensor OTensor = at::zeros({B, H, N, d}, at::kFloat);

  std::vector<float> O = formatTensor(OTensor);
  std::vector<float> Q = formatTensor(QTensor);
  std::vector<float> K = formatTensor(KTensor);
  std::vector<float> V = formatTensor(VTensor);

  const int Tr = (N + Br - 1) / Br;
  const int Tc = (N + Bc - 1) / Bc;

  for (int b = 0; b < B; b++) {
    for (int h = 0; h < H; h++) {
      std::vector<float> Sij = formatTensor(SijTensor);
      std::vector<float> Pij = formatTensor(PijTensor);
      std::vector<float> Kj = formatTensor(KjTensor);
      std::vector<float> Vj = formatTensor(VjTensor);
      std::vector<float> Qi = formatTensor(QiTensor);
      std::vector<float> Oi = formatTensor(OiTensor);
      std::vector<float> l = formatTensor(LTensor);
      std::vector<float> PV = formatTensor(PVTensor);
      std::vector<float> li = formatTensor(LiTensor);
      std::vector<float> lij = formatTensor(LijTensor);
      std::vector<float> lnew = formatTensor(LnewTensor);

      for (int j = 0; j < Tc; j++) {
        const int mx_Bc = std::min(Bc, N - j * Bc);

        for (int x = 0; x < mx_Bc; x++) {
          for (int y = 0; y < d; y++) {
            twoDimWrite(Kj, x, y, d, fourDimRead(K, b, h, j * Bc + x, y, H, N, d));
            twoDimWrite(Vj, x, y, d, fourDimRead(V, b, h, j * Bc + x, y, H, N, d));
          }
        }

        int i = 0;
        while (i < Tr) {
          const int mx_Br = std::min(Br, N - i * Br);

          for (int x = 0; x < mx_Br; x++) {
            for (int y = 0; y < d; y++) {
              twoDimWrite(Qi, x, y, d, fourDimRead(Q, b, h, i * Br + x, y, H, N, d));
              twoDimWrite(Oi, x, y, d, fourDimRead(O, b, h, i * Br + x, y, H, N, d));
              li[x] = l[i * Br + x];
            }
          }

          for (int x = 0; x < mx_Br; x++) {
            for (int y = 0; y < mx_Bc; y++) {
              float sum = 0.0;
              for (int z = 0; z < d; z++) {
                sum += twoDimRead(Qi, x, z, d) * twoDimRead(Kj, y, z, d);
              }
              twoDimWrite(Sij, x, y, Bc, sum);
            }
          }

          for (int x = 0; x < mx_Br; x++) {
            for (int y = 0; y < mx_Bc; y++) {
              twoDimWrite(Pij, x, y, Bc, std::exp(twoDimRead(Sij, x, y, Bc)));
            }
          }

          for (int x = 0; x < mx_Br; x++) {
            float sum = 0.0;
            for (int y = 0; y < mx_Bc; y++) {
              sum += twoDimRead(Pij, x, y, Bc);
            }
            lij[x] = sum;
          }

          for (int x = 0; x < mx_Br; x++) {
            lnew[x] = li[x] + lij[x];
          }

          for (int x = 0; x < mx_Br; x++) {
            for (int y = 0; y < d; y++) {
              float sum = 0.0;
              for (int z = 0; z < mx_Bc; z++) {
                sum += twoDimRead(Pij, x, z, Bc) * twoDimRead(Vj, z, y, d);
              }
              twoDimWrite(Oi, x, y, d,
                          (li[x] * twoDimRead(Oi, x, y, d) + sum) / lnew[x]);
            }
          }

          for (int x = 0; x < mx_Br; x++) {
            for (int y = 0; y < d; y++) {
              fourDimWrite(O, b, h, i * Br + x, y, H, N, d,
                           twoDimRead(Oi, x, y, d));
            }
            l[i * Br + x] = lnew[x];
          }

          i++;  // while loop increment
        }
      }
    }
  }

  return torch::from_blob(O.data(), {B, H, N, d},
                          torch::TensorOptions().dtype(torch::kFloat32))
      .clone();
}

/* DO NOT EDIT THESE BINDINGS */
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("myNaiveAttention", &myNaiveAttention, "Naive Attention");
  m.def("myUnfusedAttentionBlocked", &myUnfusedAttentionBlocked,
        " Blocked Unfused Attention");
  m.def("myFusedAttention", &myFusedAttention, "Fused Attention");
  m.def("myFlashAttention", &myFlashAttention, "Flash Attention");
  m.def("twoDimRead", &twoDimRead, "twoDimRead");
  m.def("fourDimRead", &fourDimRead, "fourDimRead");
}
