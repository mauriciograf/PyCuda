import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.compiler import SourceModule

# Definindo o kernel CUDA que vai somar dois vetores
mod = SourceModule("""
__global__ void vector_add(float *a, float *b, float *c, int N)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        c[idx] = a[idx] + b[idx];
    }
}
""")

# Função para somar vetores com PyCUDA
def vector_addition():
    # Tamanho dos vetores
    N = 100000  # Número de elementos nos vetores

    # Criando dois vetores de entrada (a e b) com números aleatórios
    a = np.random.rand(N).astype(np.float32)
    b = np.random.rand(N).astype(np.float32)

    # Vetor de saída (c) que armazenará a soma
    c = np.zeros_like(a)

    # Alocando memória na GPU
    a_gpu = cuda.mem_alloc(a.nbytes)
    b_gpu = cuda.mem_alloc(b.nbytes)
    c_gpu = cuda.mem_alloc(c.nbytes)

    # Transferindo dados da CPU para a GPU
    cuda.memcpy_htod(a_gpu, a)
    cuda.memcpy_htod(b_gpu, b)

    # Obtendo o kernel CUDA da fonte
    vector_add = mod.get_function("vector_add")

    # Definindo a quantidade de blocos e threads
    block_size = 256  # Tamanho do bloco
    grid_size = (N + block_size - 1) // block_size  # Número de blocos

    # Executando o kernel
    vector_add(a_gpu, b_gpu, c_gpu, np.int32(N), block=(block_size, 1, 1), grid=(grid_size, 1))

    # Copiando o resultado da GPU para a CPU
    cuda.memcpy_dtoh(c, c_gpu)

    # Verificando o resultado
    return c

# Executando a soma dos vetores
result = vector_addition()

# Mostrando uma parte do resultado para conferir
print("Resultado da soma dos vetores (primeiros 10 elementos):")
print(result[:10])
