# -*- coding: utf-8 -*-
"""
Вариант 10. Полный прямой проход мини-трансформера
"We eat pizza" -> "Мы едим пиццу"
d_model = 4, h = 2, d_k = 2
Простая учебная реализация на numpy (без обучения, только forward + Loss).
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)
np.random.seed(10)          # вариант 10 -> фиксируем случайные веса FFN

# ------------------------------------------------------------------
# 0. ГИПЕРПАРАМЕТРЫ
# ------------------------------------------------------------------
d_model = 4      # размер вектора
h = 2      # число голов
d_k = 2      # размер одной головы (h * d_k = d_model)
d_ff = 8      # скрытый слой FFN

# ------------------------------------------------------------------
# 1. ЭМБЕДДИНГИ
# ------------------------------------------------------------------
src_tokens = ["We", "eat", "pizza", "<EOS>"]
E_src = np.array([
    [0.5,  0.8, -0.3, 0.1],   # We
    [1.2,  0.5,  0.6, 0.9],   # eat
    [0.9, -0.1, -1.1, 0.5],   # pizza
    [0.5,  0.5,  0.5, 0.5],   # <EOS>
])

tgt_vocab = ["<SOS>", "Мы", "едим", "пиццу", "<EOS>"]
E_tgt = np.array([
    [0.0,  0.0,  0.0, 0.0],    # <SOS>
    [0.4,  0.7, -0.2, 0.1],    # Мы
    [1.3,  0.5,  0.7, 1.0],    # едим
    [0.8, -0.1, -1.0, 0.6],    # пиццу
    [1.0,  1.0,  1.0, 1.0],    # <EOS>
])

# вход декодера (teacher forcing): <SOS> Мы едим пиццу
dec_in_ids = [0, 1, 2, 3]
# что нужно предсказать:            Мы едим пиццу <EOS>
target_ids = [1, 2, 3, 4]


# ------------------------------------------------------------------
# 2. ПОЗИЦИОННОЕ КОДИРОВАНИЕ
# ------------------------------------------------------------------
def positional_encoding(n_pos, d):
    PE = np.zeros((n_pos, d))
    for pos in range(n_pos):
        for i in range(0, d, 2):
            w = 1.0 / (10000 ** (i / d))
            PE[pos, i] = np.sin(pos * w)
            PE[pos, i + 1] = np.cos(pos * w)
    return PE


# ------------------------------------------------------------------
# 3. БАЗОВЫЕ ФУНКЦИИ
# ------------------------------------------------------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def layer_norm(x, eps=1e-6):
    mu = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    return (x - mu) / (sd + eps)


def attention(Q, K, V, mask=None):
    """Scaled Dot-Product Attention."""
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    A = softmax(scores)
    return A @ V, A


def multi_head(X_q, X_kv, Wq, Wk, Wv, Wo, mask=None, name=""):
    """Многоголовое внимание: h голов по d_k, затем конкатенация."""
    Q = X_q  @ Wq
    K = X_kv @ Wk
    V = X_kv @ Wv

    heads = []
    for i in range(h):
        s = slice(i * d_k, (i + 1) * d_k)          # срез головы
        head, A = attention(Q[:, s], K[:, s], V[:, s], mask)
        heads.append(head)
        if name:
            print(f"   {name} | голова {i + 1}: веса внимания\n{A}")
    return np.concatenate(heads, axis=1) @ Wo


def ffn(X, W1, b1, W2, b2):
    return np.maximum(0, X @ W1 + b1) @ W2 + b2     # ReLU


# ------------------------------------------------------------------
# 4. ВЕСА МОДЕЛИ
# ------------------------------------------------------------------
I = np.eye(d_model)                 # для наглядности W_Q=W_K=W_V=W_O=I
Wq = Wk = Wv = Wo = I


def rand_ffn():
    W1 = np.round(np.random.uniform(-0.5, 0.5, (d_model, d_ff)), 2)
    b1 = np.zeros(d_ff)
    W2 = np.round(np.random.uniform(-0.5, 0.5, (d_ff, d_model)), 2)
    b2 = np.zeros(d_model)
    return W1, b1, W2, b2


enc_ffn = rand_ffn()
dec_ffn = rand_ffn()

# выходной слой: связанные веса (logits = h · E_tgt^T)
W_out = E_tgt.T
b_out = np.zeros(len(tgt_vocab))


# ------------------------------------------------------------------
# ШАГ 1. X_src и X_tgt
# ------------------------------------------------------------------
print("=" * 62)
print("ШАГ 1. Эмбеддинги + позиционное кодирование")
print("=" * 62)

X_src = E_src + positional_encoding(len(src_tokens), d_model)
X_tgt = E_tgt[dec_in_ids] + positional_encoding(len(dec_in_ids), d_model)

print("X_src =\n", X_src)
print("X_tgt =\n", X_tgt)


# ------------------------------------------------------------------
# ШАГ 2. ЭНКОДЕР
# ------------------------------------------------------------------
print("\n" + "=" * 62)
print("ШАГ 2. Энкодер")
print("=" * 62)

attn_enc = multi_head(X_src, X_src, Wq, Wk, Wv, Wo, name="Self-Attn энкодера")
Z1 = layer_norm(X_src + attn_enc)                 # Add & Norm
Z2 = layer_norm(Z1 + ffn(Z1, *enc_ffn))           # Add & Norm
X_out_enc = Z2

print("После Add&Norm 1 =\n", Z1)
print("X_out_enc =\n", X_out_enc)


# ------------------------------------------------------------------
# ШАГ 3. ДЕКОДЕР
# ------------------------------------------------------------------
print("\n" + "=" * 62)
print("ШАГ 3. Декодер (позиции 0,1,2,3)")
print("=" * 62)

n = len(dec_in_ids)
causal_mask = np.tril(np.ones((n, n), dtype=bool))  # маска не смотреть вперёд

self_attn = multi_head(X_tgt, X_tgt, Wq, Wk, Wv, Wo,
                       mask=causal_mask, name="Masked Self-Attn")
D1 = layer_norm(X_tgt + self_attn)

cross_attn = multi_head(D1, X_out_enc, Wq, Wk, Wv, Wo, name="Cross-Attn")
D2 = layer_norm(D1 + cross_attn)

X_out_dec = layer_norm(D2 + ffn(D2, *dec_ffn))
print("X_out_dec =\n", X_out_dec)


# ------------------------------------------------------------------
# ШАГ 4-5. ЛОГИТЫ, SOFTMAX, LOSS
# ------------------------------------------------------------------
print("\n" + "=" * 62)
print("ШАГ 4. Логиты, Softmax и Loss по позициям")
print("=" * 62)

logits = X_out_dec @ W_out + b_out
probs = softmax(logits)

losses = []
for i in range(n):
    t = target_ids[i]
    loss_i = -np.log(probs[i, t])
    losses.append(loss_i)

    print(f"\nПозиция {i}: вход '{tgt_vocab[dec_in_ids[i]]}' "
          f"-> цель '{tgt_vocab[t]}'")
    print("  Logits :", logits[i])
    print("  Softmax:", probs[i])
    for w, p in zip(tgt_vocab, probs[i]):
        print(f"      P({w:6s}) = {p:.4f}")
    print(f"  Предсказано: '{tgt_vocab[np.argmax(probs[i])]}'")
    print(f"  Loss_{i} = -ln({probs[i, t]:.4f}) = {loss_i:.4f}")

mean_loss = float(np.mean(losses))

print("\n" + "=" * 62)
print("ШАГ 5. Средний Loss")
print("=" * 62)
print("(" + " + ".join(f"{l:.4f}" for l in losses) + ") / 4")
print(f"Средний Loss = {mean_loss:.4f}")
print(f"Perplexity   = {np.exp(mean_loss):.4f}")
