from OverlapCode import overlap_save
import threading
import numpy as np
import sounddevice as sd
from pynput import keyboard
import librosa
import pysofaconventions as sofa

'''funcionalidade do codigo: rodar o audio enquanto e possivel alterar a direcao da fonte sonora
atraves das setas do teclado'''

# config
SOFA_PATH = r"C:\Users\m\Downloads\KEMAR_GRAS_EarSim_SmallEars_FreeFieldComp_44kHz.sofa"
AUDIO_PATH = r"C:\Users\m\Downloads\260cde50.wav"

STEP_DEG = 10              # passo em graus por seta
N_FADE_BLOCKS = 20         # crossfade ao trocar HRTF 
N = 4096                   # tamanho do bloco processado


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# abrir banco de dados
try:
    sofa_data = sofa.SOFAFile(SOFA_PATH, 'r')
except Exception as e:
    raise SystemExit(f"Erro ao abrir o arquivo SOFA: {e}")

source_pos = sofa_data.getVariableValue("SourcePosition")  # (M, 3) normalmente [az, el, r]
ir_all = sofa_data.getDataIR()  # (Mpos, 2, Nir)
Mpos, Rears, Nir = ir_all.shape
if Rears != 2:
    raise SystemExit(f"Esperava 2 ouvidos (R=2), encontrei R={Rears}")

# fs sofa
fs_raw = sofa_data.getSamplingRate()
fs_sofa = int(fs_raw.flatten()[0] if isinstance(fs_raw, np.ndarray) else fs_raw)

print(f"SOFA: {Mpos} posições; IR len={Nir}; fs={fs_sofa} Hz")

# audio
audio_mono, _ = librosa.load(AUDIO_PATH, sr=fs_sofa, mono=True)
print(f"Áudio mono: {len(audio_mono)/fs_sofa:.2f}s @ {fs_sofa} Hz")

# definir n/l
# L = N - M + 1 uteis por bloco
M_ir = Nir

if N < M_ir:
    N = 1 << (int((M_ir - 1)).bit_length())  # ajusta para power-of-two >= M
L = N - M_ir + 1
print(f"N={N}, M={M_ir}, L (úteis/bloco)={L}, latência ≈ {L/fs_sofa*1000:.1f} ms")

# lookup posicao
def get_index_nearest(az_deg: float, el_deg: float) -> int:
    d = (source_pos[:, 0] - az_deg) ** 2 + (source_pos[:, 1] - el_deg) ** 2
    return int(np.argmin(d))

def get_ir_pair(ind: int):
    pair = ir_all[ind]  # (2, Nir)
    return pair[0].astype(np.float32), pair[1].astype(np.float32)

# estado interativo
state_lock = threading.Lock()
desired_az = 45.0
desired_el = 45.0
pending_filter_update = True
stop_flag = False

# tails do sinal de entrada
tail_input = np.zeros(M_ir - 1, dtype=np.float32)

# crossfade
fade_blocks_left = 0
HRTF_prev = None  # tuple (irL_prev, irR_prev)
HRTF_cur = None   # tuple (irL_cur, irR_cur)

# posição corrente (info)
cur_idx = None
cur_pos = None

# load hrtf (funcao de transferencia)
def load_hrtf_for(az, el):
    idx = get_index_nearest(az, el)
    irL, irR = get_ir_pair(idx)
    used_pos = source_pos[idx]
    return (irL, irR), idx, used_pos

HRTF_cur, cur_idx, cur_pos = load_hrtf_for(desired_az, desired_el)
print(f"Inicial: idx={cur_idx}, pos=[az={cur_pos[0]:.1f}, el={cur_pos[1]:.1f}]")

# callback audio
play_cursor = 0

def process_chunk_with_ir(x_chunk: np.ndarray, irL: np.ndarray, irR: np.ndarray,
                          tail_in: np.ndarray):
    """
    Usa SUA overlap_save assim:
      - Monta x_state = [tail_in] + [x_chunk]  (len = (M-1) + L)
      - y_state = overlap_save(x_state, h, N)
      - Retorna as últimas L amostras (válidas) para cada canal
      - Atualiza e devolve tail_in = últimos (M-1) de x_state
    """
    x_state = np.concatenate([tail_in, x_chunk]).astype(np.float32)

    # esquerdo
    yL_state = overlap_save(x_state, irL.astype(np.float32), N)
    yL = yL_state[-len(x_chunk):]  # pega só as últimas L

    # direito
    yR_state = overlap_save(x_state, irR.astype(np.float32), N)
    yR = yR_state[-len(x_chunk):]

    new_tail_in = x_state[-(M_ir - 1):] if M_ir > 1 else np.zeros(0, dtype=np.float32)
    return yL, yR, new_tail_in

def audio_callback(outdata, frames, time, status):
    global play_cursor, pending_filter_update, cur_idx, cur_pos
    global tail_input, fade_blocks_left, HRTF_prev, HRTF_cur, stop_flag

    if status:
        print("Status:", status)

    # garante que frames == L
    if frames != L:
        # preenche silêncio se der mismatch (não deveria acontecer com blocksize=L)
        outdata[:] = 0.0
        return

    # atualizacao hrtf
    with state_lock:
        if pending_filter_update:
            new_hrtf, new_idx, new_pos = load_hrtf_for(desired_az, desired_el)
            HRTF_prev = HRTF_cur
            HRTF_cur = new_hrtf
            cur_idx, cur_pos = new_idx, new_pos
            fade_blocks_left = max(1, int(N_FADE_BLOCKS))
            pending_filter_update = False
            print(f">>> Novo HRTF: idx={cur_idx}, pos=[az={cur_pos[0]:.1f}, el={cur_pos[1]:.1f}]")

    # obter chunck audio
    if play_cursor + L > len(audio_mono):
        rem = len(audio_mono) - play_cursor
        if rem > 0:
            x_chunk = np.zeros(L, dtype=np.float32)
            x_chunk[:rem] = audio_mono[play_cursor:play_cursor + rem].astype(np.float32)
        else:
            x_chunk = np.zeros(L, dtype=np.float32)
            stop_flag = True
    else:
        x_chunk = audio_mono[play_cursor:play_cursor + L].astype(np.float32)

    play_cursor += L

    # processar com hrtf atual
    yL_cur, yR_cur, tail_next = process_chunk_with_ir(x_chunk, HRTF_cur[0], HRTF_cur[1], tail_input)

    if fade_blocks_left > 0 and HRTF_prev is not None:
        # tambem processa com HRTF anterior (usa a MESMA tail de entrada para manter alinhamento)
        yL_prev, yR_prev, _ = process_chunk_with_ir(x_chunk, HRTF_prev[0], HRTF_prev[1], tail_input)
        alpha = 1.0 - (fade_blocks_left / max(1, N_FADE_BLOCKS))
        yL = (1 - alpha) * yL_prev + alpha * yL_cur
        yR = (1 - alpha) * yR_prev + alpha * yR_cur
        fade_blocks_left -= 1
        if fade_blocks_left == 0:
            HRTF_prev = None  # libera
    else:
        yL, yR = yL_cur, yR_cur

    # atualiza a tail de entrada (com base no processamento atual)
    tail_input = tail_next

    # stereo out
    y_st = np.stack([yL, yR], axis=1).astype(np.float32)
    outdata[:] = y_st

# codigo para o teclado, alterar percepcao com as setas
def on_press(key):
    global desired_az, desired_el, pending_filter_update, stop_flag
    try:
        if key == keyboard.Key.right:
            with state_lock:
                desired_az += STEP_DEG
                pending_filter_update = True
        elif key == keyboard.Key.left:
            with state_lock:
                desired_az -= STEP_DEG
                pending_filter_update = True
        elif key == keyboard.Key.up:
            with state_lock:
                desired_el = clamp(desired_el + STEP_DEG, -90.0, 90.0)
                pending_filter_update = True
        elif key == keyboard.Key.down:
            with state_lock:
                desired_el = clamp(desired_el - STEP_DEG, -90.0, 90.0)
                pending_filter_update = True
        elif hasattr(key, 'char') and key.char in ('r', 'R'):
            with state_lock:
                desired_az = 0.0
                desired_el = 0.0
                pending_filter_update = True
        elif hasattr(key, 'char') and key.char in ('q', 'Q'):
            stop_flag = True
            return False
    except Exception:
        pass

listener = keyboard.Listener(on_press=on_press)
listener.start()

print("\nControles: ←/→ azimute ±10°, ↑/↓ elevação ±10°, R reset (0,0), Q sair\n")


# fixar blocksize = L para ser compativel com o overlap save
stream = sd.OutputStream(
    samplerate=fs_sofa,
    channels=2,
    dtype='float32',
    blocksize=L,
    callback=audio_callback
)

with stream:
    try:
        while not stop_flag:
            sd.sleep(50)
    except KeyboardInterrupt:
        pass

listener.stop()
print("Finalizado.")
