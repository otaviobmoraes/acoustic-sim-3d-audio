import numpy as np
import pysofaconventions as sofa
from scipy.io.wavfile import read, write
from OverlapCode import overlap_save
import sounddevice as sd
import librosa


'''funcionalidade do codigo: definir o parametro de direcao e rodar o audio que sera direcionado
dos parametros escolhidos.'''

SAMPLE_RATE = 44100

# leitura do arquivo SOFA

sofa_file_path = r"C:\Users\m\Downloads\KEMAR_GRAS_EarSim_SmallEars_FreeFieldComp_44kHz.sofa"

# definir angulacao
azimute_alvo = -45
elevacao_alvo = 45


# abrir arquivo
try:
    sofa_data = sofa.SOFAFile(sofa_file_path, 'r')
except Exception as e:
    print(f"Erro ao abrir o arquivo SOFA: {e}")
    exit()

# obter posicoes
source_pos = sofa_data.getVariableValue("SourcePosition")  # (M, 3)

# achar indice proximo ao desejado
distancias = np.sqrt(
    (source_pos[:, 0] - azimute_alvo) ** 2 +
    (source_pos[:, 1] - elevacao_alvo) ** 2
)

indice = np.argmin(distancias)

print(f"Índice mais próximo encontrado: {indice}")
print(f"Posição real usada: {source_pos[indice]}")

# obter a resposta a angulacao desejada
ir = sofa_data.getDataIR()
ir_angulacao = ir[indice, :, :]

# salvar em respostas separadas
ir_left = ir_angulacao[0, :]
ir_right = ir_angulacao[1, :]

print("\nAs variáveis 'ir_left' e 'ir_right' foram salvas e estão prontas para serem usadas.")
print(f"Tamanho da IR do ouvido esquerdo: {len(ir_left)} amostras")
print(f"Tamanho da IR do ouvido direito: {len(ir_right)} amostras")

fs_raw = sofa_data.getSamplingRate()
if isinstance(fs_raw, np.ndarray):
    fs = int(fs_raw.flatten()[0])
else:
    fs = int(fs_raw)

print(f"Taxa de amostragem (fs) extraída e corrigida para: {fs} Hz")

# processamento do audio

audio_file_path = "C:\\Users\\m\\Downloads\\260cde50.wav"

try:
    audio_mono, fs_audio = librosa.load(audio_file_path, sr=fs, mono=True)
    print(f"Áudio mono carregado com sucesso. Taxa de amostragem: {fs_audio} Hz")
    
except Exception as e:
    print(f"Ocorreu um erro ao carregar o arquivo de áudio: {e}")
    exit()

# definicao do bloco
M = len(ir_left)
N = 2048
if N < M:
    raise ValueError(f"O tamanho do bloco (N={N}) deve ser maior que o tamanho do filtro (M={M}).")

# aplicar o metodo overlapsave

print("\nIniciando convolução...")
audio_saida_left = overlap_save(audio_mono, ir_left, N)
audio_saida_right = overlap_save(audio_mono, ir_right, N)
print("Convolução finalizada.")

# formar audio estereo
audio_estéreo = np.vstack([audio_saida_left, audio_saida_right]).T

audio_estéreo_int16 = (audio_estéreo * 32767).astype(np.int16)

output_file_path = "audio_espacializado.wav"
write(output_file_path, fs, audio_estéreo_int16)

print(f"\nÁudio estéreo espacializado salvo em: {output_file_path}")

try:
    fs_output, audio_stereo_int16 = read(output_file_path)

    if fs_output != SAMPLE_RATE:
        print(f"Atenção: A taxa de amostragem do arquivo ({fs_output} Hz) é diferente da esperada ({SAMPLE_RATE} Hz).")
        print("A reprodução pode ter uma velocidade incorreta.")

    audio_stereo_float = audio_stereo_int16.astype(np.float32) / 32768.0

    print(f"Reproduzindo arquivo: {output_file_path}")
    print(f"Taxa de amostragem: {fs_output} Hz")
    print(f"Duração: {len(audio_stereo_float) / fs_output:.2f} segundos")

    #tocar audio

    sd.play(audio_stereo_float, samplerate=fs_output)
    sd.wait()

    print("Reprodução finalizada.")

except FileNotFoundError:
    print(f"Erro: O arquivo de áudio não foi encontrado em '{output_file_path}'.")
    print("Por favor, verifique se o script de convolução foi executado e o arquivo foi salvo corretamente.")
except Exception as e:
    print(f"Ocorreu um erro durante a reprodução: {e}")