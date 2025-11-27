# Монгол-Англи Хэл Орчуулагч

PyTorch дээр суурилсан Transformer архитектур ашиглан монгол хэлнээс англи хэл рүү орчуулах neural machine translation систем. Мөн англи хэлээр дамжуулан герман болон испани хэл рүү орчуулах боломжтой.

## Онцлог шинж чанарууд

- **Transformer архитектур**: Encoder-decoder бүтэцтэй multi-head attention механизм
- **BERT токенизатор**: Монгол болон англи хэлний урьдчилан сургагдсан BERT токенизатор ашигласан
- **Олон хэл дэмжих**: MarianMT загваруудаар дамжуулан герман, испани хэл рүү орчуулах
- **WER үнэлгээ**: Орчуулгын чанарыг Word Error Rate-ээр хэмжих
- **Positional encoding**: Өөрийн гэсэн sinusoidal positional encoding алгоритм


## Өгөгдлийн формат

Сургалтын өгөгдөл нь монгол болон англи хэлний хос өгүүлбэрүүдийг агуулсан файл байх ёстой:

```
Монгол өгүүлбэр+++++SEP+++++English sentence
```

Жишээ:
```
Би ном унших дуртай.+++++SEP+++++I like reading books.
```

## Загварын бүтэц

- **Embedding хэмжээ**: 256
- **Attention heads**: 8
- **Encoder/Decoder давхаргууд**: 3 давхарга тус бүр
- **Feedforward хэмжээ**: 512
- **Максимум урт**: 40 токен
- **Dropout**: 0.1

## Хэрэглээ

### Сургалт

```python
# Өгөгдөл уншиж бэлтгэх
examples = []
with codecs.open("mn_en.txt", 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        arr = line.strip().split("+++++SEP+++++")
        if len(arr) == 2:
            examples.append((arr[0].strip(), arr[1].strip()))

# Загварыг сургах
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = evaluate(model, val_loader, criterion, device)
```

### Орчуулга

```python
# Монгол → Англи
mongolian_sentence = "Би ном унших дуртай."
translated = translate_sentence(model, mongolian_sentence, tokenizer_mn, tokenizer_en, MAX_LENGTH, device)
print(f"Орчуулга: {translated}")
```

### Олон хэл рүү орчуулах

```python
# Монгол → Англи → Герман/Испани
en_translation = translate_sentence(model, mn_sentence, tokenizer_mn, tokenizer_en, MAX_LENGTH, device)
de_translation = translate_marian(marian_model_de, marian_tokenizer_de, en_translation, MAX_LENGTH, device)
es_translation = translate_marian(marian_model_es, marian_tokenizer_es, en_translation, MAX_LENGTH, device)
```

## Урьдчилан сургагдсан загварууд

Системд дараах загваруудыг ашигласан:

- **Монгол BERT**: `tugstugi/bert-base-mongolian-cased`
- **Англи BERT**: `bert-base-uncased`
- **Герман орчуулга**: `Helsinki-NLP/opus-mt-en-de`
- **Испани орчуулга**: `Helsinki-NLP/opus-mt-en-es`

## Үнэлгээ

Загварын чанарыг Word Error Rate (WER)-ээр үнэлнэ:

```python
from jiwer import wer

reference = "i like reading books."
hypothesis = translate_sentence(model, mongolian_text, tokenizer_mn, tokenizer_en, MAX_LENGTH, device)
score = wer(reference, hypothesis)
```

## Орчуулгын жишээнүүд

| Монгол | Англи | Герман | Испани |
|--------|-------|--------|--------|
| Өнөөдөр сайхан өдөр байна. | Today is a beautiful day. | Heute ist ein schöner Tag. | Hoy es un día hermoso. |
| Би зүгээр ээ, баярлалаа. | I'm fine, thank you. | Mir geht's gut, danke. | Estoy bien, gracias. |
| Би аз жаргалтай байна. | I am happy. | Ich bin glücklich. | Estoy feliz. |

## Сургалтын тохиргоо
- **Batch size**: 32
- **Learning rate**: 0.0001
- **Optimizer**: Adam
- **Loss функц**: CrossEntropyLoss (padding токенуудыг орхисон)
- **Train/Validation хуваарь**: 80/20

## Текст боловсруулалт

Систем нь дараах боловсруулалтуудыг хийнэ:
- Товчлолуудыг задлах (жишээ нь: "don't" → "do not")
- Тэмдэглэгээ арилгах
- Жижиг үсэг болгох
- Токенизацын алдаануудыг засах
