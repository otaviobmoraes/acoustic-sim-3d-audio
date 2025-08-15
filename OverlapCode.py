import numpy as np
from scipy.fft import fft, fftfreq

def overlap_save(x, h, N):
    """
    Aplica convolução circular entre x[n] e h[n] usando o método overlap-save com FFT.

    Parâmetros:
        x : np.ndarray
            Sinal de entrada (1D).
        h : np.ndarray
            Filtro FIR (1D).
        N : int
            Tamanho do bloco (deve ser >= len(h)).

    Retorno:
        y : np.ndarray
            Sinal de saída filtrado.
    """
    M = len(h) # tamanho do filtro
    L = N - M + 1  # numero de amostras uteis por bloco
    
    # transformada de fourier do filtro (funcao de transferencia)
    H = np.fft.fft(h, n=N)

    # preencher x com zeros no inicio para o primeiro bloco
    x_padded = np.concatenate((np.zeros(M - 1), x)) # cada bloco de entrada deve ter M - 1 amostras anteriores, preenchelas com 0
    # calculo do numero de blocos necessarios para processar o sinal inteiro
    num_blocks = int(np.ceil((len(x) + M - 1) / L)) # np.ceil arredonda para cima, garantindo o numero suficiente de blocos

    y = []

    for i in range(num_blocks):
        # extrair bloco e retirar parte m - 1
        start = i * L
        end = start + N
        x_block = x_padded[start:end]

        # se x for menor que N, fazer zero-padding para corrigir, completando com zeros no final, pode ocorrer no ultimo bloco do sinal
        if len(x_block) < N:
            x_block = np.pad(x_block, (0, N - len(x_block)))

        # transformada de fourier do bloco para ir para dominio da frequencia
        X = np.fft.fft(x_block)
        
        # convolucao circular no dominio da frequencia
        Y = X * H

        # transformada de fourier inversa para voltar ao dominio do tempo
        y_block = np.fft.ifft(Y).real

        # salvar apenas as ultimas L amostras (parte valida)
        y.extend(y_block[M - 1:])
        

    return np.array(y[:len(x)])    # garantir que a saida tenha o mesmo tamanho da entrada
