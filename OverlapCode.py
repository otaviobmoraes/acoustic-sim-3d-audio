import numpy as np

import matplotlib.pyplot as plt

from scipy.fft import fft, fftfreq

import sounddevice as sd

import pysofaconventions

#codigo opcional, mostra o dispositivos de saida e entrada
#print(sd.query_devices())

'''
SAMPLE_RATE   = 44100  #hz frequencia padrao, achei na internet
CHANNELS = 1 # defini o canal
FRAMES = 44100*3# define o tamanho da amostra

print("Gravando...")
buffer = sd.rec(frames=FRAMES,
                     samplerate=SAMPLE_RATE,
                     channels=CHANNELS,
                     dtype='float32')
sd.wait()
print("Audio gravado")

'''
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

#sd.play(buffer, samplerate=SAMPLE_RATE)
#sd.wait()


'''exemplo de uso
fs = 44100  # frequência de amostragem
t = np.arange(FRAMES) / SAMPLE_RATE

#sinal de entrada capturado pelo microfone
#x = buffer.flatten() # flatten transforma 2d para 1d

# filtro passa-baixas FIR simples 
M = 129
h = np.ones(M) / M  # filtro de média móvel

# aplicar filtro com overlap-save
N = 512  # tamanho do bloco
y = overlap_save(x, h, N)

# plotando
plt.figure(figsize=(10, 4))
plt.plot(t, x, label='Sinal Original', alpha=0.6)
plt.plot(t, y, label='Sinal Filtrado', linewidth=2)
plt.legend()
plt.title('Filtragem com Overlap-Save')
plt.xlabel('Tempo (s)')
plt.tight_layout()
plt.show()
'''

