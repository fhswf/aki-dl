# %% [markdown]
# # Ergänzung zu Termin 6: Encoder-Decoder-Netz für Bildunterschriften
#
# In der Aufgabe zu Termin 6 erzeugt ein Netz zu einem Foto aus dem COCO-Datensatz einen
# englischen Satz wie "a person riding a wave on top of a surfboard". Der Code ist fast
# vollständig vorgegeben: Ein vortrainiertes ResNeXt-101 fasst das Bild in einem Vektor zusammen,
# eine GRU erzeugt daraus Wort für Wort die Beschreibung. Ihre Aufgabe ist die letzte Zelle, in
# der Sie mit `sample_beam_search` eine Beschreibung erzeugen und mit `clean_sentence` lesbar
# machen. Um diese Zeilen schreiben und die Ergebnisse beurteilen zu können, muss man den
# übrigen Code verstehen. Kapitel 8 des Buchs liefert dafür die Residual-Verbindungen von
# ResNet, die Verwendung vortrainierter Netze als Merkmalsextraktor und die Idee der
# Encoder-Decoder-Architektur. Beam Search kommt im Buch nicht vor, wohl aber in den Folien.
#
# ```
# Foto (3, 224, 224)
#    │  ResNeXt-101 ohne letzte Schicht, eingefroren      Abschnitte 2 bis 4
#    ▼
# 2048 Werte je Bild
#    │  Linear + BatchNorm1d (EncoderCNN)                 Abschnitt 4
#    ▼
# Bildvektor mit embed_size Werten
#    │  GRU-Decoder, im Training mit der echten           Abschnitte 1, 5 und 6
#    │  Beschreibung als Eingabe
#    ▼
# je Position eine Bewertung für jedes Wort im Vokabular
#    │  Wort für Wort erzeugen: sample (greedy)           Abschnitt 7
#    │  oder sample_beam_search                           Abschnitt 8, letzte Zelle der Aufgabe
#    ▼
# Wortindizes ──► clean_sentence ──► "a person riding a wave ..."
#    Bewertung mit CIDEr                                  Abschnitt 9
# ```
#
# COCO umfasst rund 20 GB Bilder, die Gewichte von ResNeXt-101 rund 340 MB, und ein Training
# auf COCO dauert auch mit GPU Stunden. Wir arbeiten deshalb mit kleinen, selbst erzeugten
# Bildern und einem kleinen Netz, das genauso aufgebaut ist. Der Decoder ist unverändert die
# Klasse `DecoderRNN` aus der Aufgabe. Nach einem Neustart des Kernels führen Sie das Notebook
# von oben aus; alle Zellen zusammen laufen auf der CPU in wenigen Minuten.

# %% [markdown]
# ## 0. Setup

# %%
try:
    import torch  # torch ist auf Colab und dem Jupyter-Server meist schon installiert
    import torchvision
    print(f"PyTorch {torch.__version__} und torchvision {torchvision.__version__} bereits vorhanden.")
except ImportError:
    !pip install -q torch torchvision

# matplotlib für Bilder und Diagramme, tqdm für den Fortschrittsbalken beim Training
!pip install -q matplotlib==3.11.2 tqdm==4.70.1

# %% [markdown]
# ## 1. Ersatzbilder, Beschreibungen und Vokabular
#
# Jedes Ersatzbild ist 32 × 32 Pixel groß und zeigt zwei Formen: einen Kreis, ein Quadrat oder
# ein Dreieck, jeweils rot, grün, blau oder gelb, an zufälliger Stelle und in zufälliger Größe.
# Wie in COCO gehören zu jedem Bild fünf Beschreibungen, die dasselbe mit verschiedenen Worten
# sagen. Die Satzmuster kommen unterschiedlich oft vor: "there is a ... and a ..." in 40 % der
# Beschreibungen, die vier Muster "a ... and a ...", "a ... next to a ...", "a ... with a ..."
# und "a ... beside a ..." in je 15 %. Welche Form zuerst genannt wird, ist zufällig. Warum die
# Muster so verteilt sind, zeigt sich in Abschnitt 8.

# %%
import math
import random

import matplotlib.pyplot as plt
import torch

IMAGE_SIZE = 32
COLORS = {"red": (0.9, 0.15, 0.15), "green": (0.15, 0.75, 0.2),
          "blue": (0.2, 0.3, 0.9), "yellow": (0.95, 0.85, 0.1)}
SHAPES = ["circle", "square", "triangle"]
TEMPLATES = [("there is a {} and a {}", 0.4), ("a {} and a {}", 0.15), ("a {} next to a {}", 0.15),
             ("a {} with a {}", 0.15), ("a {} beside a {}", 0.15)]

yy, xx = torch.meshgrid(torch.arange(IMAGE_SIZE).float(), torch.arange(IMAGE_SIZE).float(),
                        indexing="ij")


def shape_mask(shape, cx, cy, r):
    # True für alle Pixel, die zur Form mit Mittelpunkt (cx, cy) und Radius r gehören
    if shape == "circle":
        return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    if shape == "square":
        return ((xx - cx).abs() <= 0.85 * r) & ((yy - cy).abs() <= 0.85 * r)
    return (yy <= cy + 0.8 * r) & (yy >= cy - r) & ((xx - cx).abs() <= 0.6 * (yy - cy + r))


def draw(objects):
    image = 0.05 + 0.03 * torch.randn(3, IMAGE_SIZE, IMAGE_SIZE)  # dunkler, leicht verrauschter Hintergrund
    for color, shape, cx, cy, r in objects:
        m = shape_mask(shape, cx, cy, r)
        for channel in range(3):
            image[channel][m] = COLORS[color][channel]
    return image.clamp(0, 1)


def random_places(n):
    # n Positionen mit Radius, die sich nicht überlappen
    while True:
        places = []
        for _ in range(n):
            r = random.uniform(4, 6.5)
            places.append((random.uniform(r, IMAGE_SIZE - r), random.uniform(r, IMAGE_SIZE - r), r))
        if n == 1 or math.dist(places[0][:2], places[1][:2]) > places[0][2] + places[1][2] + 2:
            return places


def random_object():
    return random.choice(list(COLORS)), random.choice(SHAPES)


def describe(a, b):
    # fünf Beschreibungen: Muster nach Häufigkeit gewählt, Reihenfolge der Formen zufällig
    captions = []
    for _ in range(5):
        template = random.choices([t for t, _ in TEMPLATES], [w for _, w in TEMPLATES])[0]
        first, second = (a, b) if random.random() < 0.5 else (b, a)
        captions.append(template.format(" ".join(first), " ".join(second)))
    return captions


def make_caption_images(n):
    images, all_captions = [], []
    for _ in range(n):
        a = random_object()
        b = random_object()
        while b == a:
            b = random_object()
        (p1, p2) = random_places(2)
        images.append(draw([(*a, *p1), (*b, *p2)]))
        all_captions.append(describe(a, b))
    return torch.stack(images), all_captions


random.seed(0)
torch.manual_seed(0)
train_images, train_captions = make_caption_images(3000)
val_images, val_captions = make_caption_images(300)

# Tippfehler wie in echten Annotationen: in genau drei Trainingsbeschreibungen "sqaure"
typo_count = 0
for captions in train_captions:
    for k, caption in enumerate(captions):
        if "square" in caption and typo_count < 3:
            captions[k] = caption.replace("square", "sqaure", 1)
            typo_count += 1

print(f"Trainingsbilder: {tuple(train_images.shape)} = 3000 Bilder, 3 Farbkanäle, 32 x 32 Pixel")
print(f"Validierungsbilder: {len(val_images)}")
print("Beschreibungen zu Trainingsbild 0:")
for caption in train_captions[0]:
    print("  ", caption)

fig, axes = plt.subplots(1, 6, figsize=(13, 2.6))
for ax, image, captions in zip(axes, train_images, train_captions):
    ax.imshow(image.permute(1, 2, 0))  # matplotlib erwartet (Höhe, Breite, Kanäle)
    ax.set_title(captions[0].replace(" and", "\nand").replace(" next", "\nnext")
                 .replace(" with", "\nwith").replace(" beside", "\nbeside"), fontsize=8)
    ax.axis("off")
plt.show()

# %% [markdown]
# Unter jedem Bild steht die erste seiner fünf Beschreibungen. Die fünf Beschreibungen zu Bild 0
# nennen dieselben zwei Formen, aber in wechselnder Reihenfolge und mit verschiedenen Mustern.
# Für ein Netz bedeutet das: Auf ein Bild gibt es mehrere richtige Antworten. Bei COCO ist das
# genauso, nur sind die fünf Sätze dort von Menschen frei formuliert.
#
# Aus den Beschreibungen entsteht nun das Vokabular, so wie es die Klasse `Vocabulary` der
# Aufgabe aus `captions_train2014.json` baut. Zuerst kommen vier Sondertokens mit festen
# Nummern: `<pad>` (0) zum Auffüllen, `<start>` (1) und `<end>` (2) für Anfang und Ende einer
# Beschreibung, `<unk>` (3) für Wörter, die nicht im Vokabular stehen. Danach folgen alle Wörter,
# die mindestens `vocab_threshold`-mal vorkommen; seltenere Wörter werden später zu `<unk>`. Die
# Aufgabe zerlegt die Sätze mit `nltk.tokenize.word_tokenize`, weil COCO-Beschreibungen Satzzeichen
# und Großbuchstaben enthalten. Unsere Sätze sind schon klein geschrieben und ohne Satzzeichen,
# deshalb genügt `split()`.

# %%
from collections import Counter

PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"


def build_vocab(all_captions, vocab_threshold):
    counter = Counter(word for captions in all_captions for caption in captions
                      for word in caption.split())
    idx2word = [PAD, START, END, UNK] + [w for w, n in counter.items() if n >= vocab_threshold]
    word2idx = {word: idx for idx, word in enumerate(idx2word)}
    return word2idx, idx2word, counter


vocab_threshold = 5
word2idx, idx2word, word_counts = build_vocab(train_captions, vocab_threshold)
vocab_size = len(idx2word)
print(f"vocab_size = {vocab_size}")
print("idx2word:", idx2word)
print(f"'square' kommt {word_counts['square']}-mal vor, 'sqaure' {word_counts['sqaure']}-mal")

# %% [markdown]
# Das Vokabular hat 19 Einträge: die vier Sondertokens und 15 Wörter. "sqaure" kommt nur dreimal
# vor, also seltener als `vocab_threshold = 5`, und fehlt deshalb. Alle drei Tippfehler stehen in
# den Beschreibungen von Bild 0 oben. Bei COCO kommen mit derselben Schwelle mehrere Tausend
# Wörter ins Vokabular; Tippfehler und sehr seltene Wörter bleiben auch dort draußen.
#
# Für das Training wird jede Beschreibung zu einer Folge von Wortnummern. `CocoDataset` in der
# Aufgabe setzt `<start>` davor und `<end>` dahinter und füllt mit `<pad>` auf, bis alle Folgen
# gleich lang sind, nämlich die längste Beschreibung plus 2. Außerdem gibt es die tatsächliche
# Länge `length` zurück. Die folgende Funktion macht dasselbe.

# %%
max_length = max(len(c.split()) for captions in train_captions for c in captions)


def encode(caption):
    ids = [word2idx[START]] + [word2idx.get(w, word2idx[UNK]) for w in caption.split()] + [word2idx[END]]
    length = len(ids)
    ids += [word2idx[PAD]] * (max_length + 2 - length)
    return torch.tensor(ids), length


typo_caption = next(c for captions in train_captions for c in captions if "sqaure" in c)
ids, length = encode(typo_caption)
print(f"max_length = {max_length} Wörter, also Folgen der Länge {max_length + 2}")
print(f"Beschreibung: '{typo_caption}'")
print(f"Nummern:      {ids.tolist()}   length = {length}")
print("Zurück:      ", [idx2word[i] for i in ids])

# %% [markdown]
# Die längsten Beschreibungen haben 9 Wörter, etwa "there is a red square and a yellow square";
# jede Folge hat daher 11 Einträge. Die Beispielbeschreibung mit dem Tippfehler zeigt alle
# Sondertokens: Die 1 am Anfang ist `<start>`, die 3 steht für das unbekannte "sqaure", die 2 ist
# `<end>`, und die Nullen am Ende sind `<pad>`. `length` zählt `<start>` und `<end>` mit, aber
# nicht das Padding.
#
# Zum Schluss verpacken wir Bilder und Beschreibungen in ein `Dataset`. Es ist aufgebaut wie
# `CocoDataset` im Modus `"train"`: Ein Eintrag ist eine Beschreibung, nicht ein Bild; jedes Bild
# kommt also fünfmal vor. `__getitem__` gibt wie in der Aufgabe Bild, Nummernfolge, Länge und
# eine Kennung zurück.

# %%
from torch.utils.data import DataLoader, Dataset


class ShapesCaptionDataset(Dataset):
    def __init__(self, images, all_captions):
        self.images = images
        # eine Zeile je Beschreibung: (Bildnummer, Beschreibung)
        self.entries = [(i, c) for i, captions in enumerate(all_captions) for c in captions]

    def __getitem__(self, index):
        image_idx, caption = self.entries[index]
        ids, length = encode(caption)
        return self.images[image_idx], ids, length, image_idx

    def __len__(self):
        return len(self.entries)


train_set = ShapesCaptionDataset(train_images, train_captions)
val_set = ShapesCaptionDataset(val_images, val_captions)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64)

images, captions, lengths, image_ids = next(iter(train_loader))
print(f"Einträge im Trainingsset: {len(train_set)} = 3000 Bilder x 5 Beschreibungen")
print(f"images:   {tuple(images.shape)} = 64 Bilder, 3 Kanäle, 32 x 32")
print(f"captions: {tuple(captions.shape)} = 64 Beschreibungen, je {max_length + 2} Nummern")
print(f"lengths:  {tuple(lengths.shape)}, z. B. {lengths[:6].tolist()}")

# %% [markdown]
# Ein Batch enthält 64 Bilder, 64 gleich lange Nummernfolgen und 64 Längen. Die Längen liegen
# zwischen 9 und 11 Einträgen. In der Aufgabe sehen die Batches genauso aus, nur mit Bildern der
# Größe 224 × 224 und deutlich längeren Folgen, weil die längste COCO-Beschreibung viel länger
# ist als die meisten.

# %% [markdown]
# ## 2. Warum tiefer nicht automatisch besser ist
#
# Der Encoder der Aufgabe ist ein ResNeXt-101, also ein Netz mit 101 Schichten. Das Buch
# beginnt Kapitel 8 mit der Beobachtung, dass zusätzliche Schichten ein Netz zunächst besser
# machen, ab einer gewissen Tiefe aber schwerer zu trainieren, weil die Gradienten auf dem Weg
# durch viele Schichten verschwinden oder explodieren. Zur Erinnerung: Der Gradient eines
# Gewichts gibt an, wie stark sich der Verlust ändert, wenn man dieses Gewicht ein wenig ändert.
# Ist er nahezu null, bleibt das Gewicht beim Training praktisch stehen.
#
# Das lässt sich an einem einfachen Netz aus `nn.Linear`-Schichten mit ReLU beobachten. Es soll
# aus zwei Eingabewerten eine Kurve vorhersagen. Wir vergleichen 3, 10 und 30 Schichten,
# jeweils in zwei Varianten. In der ersten gibt jede Schicht ihre Ausgabe einfach an die nächste
# weiter. In der zweiten wird die Eingabe einer Schicht zusätzlich auf ihre Ausgabe addiert; das
# ist die Abkürzungsverbindung von ResNet. Für jede Variante geben wir die Größe des Gradienten
# in der ersten Schicht aus und den Verlust nach 300 Trainingsschritten.

# %%
import torch.nn as nn
import torch.nn.functional as F


class DeepNet(nn.Module):
    def __init__(self, depth, residual, width=32):
        super().__init__()
        self.first = nn.Linear(2, width)
        self.hidden = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.last = nn.Linear(width, 1)
        self.residual = residual

    def forward(self, x):
        x = F.relu(self.first(x))
        for layer in self.hidden:
            if self.residual:
                x = F.relu(x + layer(x))  # Eingabe auf die Ausgabe addieren
            else:
                x = F.relu(layer(x))
        return self.last(x)


torch.manual_seed(0)
x_curve = torch.rand(512, 2) * 4 - 2
y_curve = torch.sin(2 * x_curve[:, :1]) + x_curve[:, 1:] ** 2

for depth in [3, 10, 30]:
    for residual in [False, True]:
        torch.manual_seed(1)
        net = DeepNet(depth, residual)
        F.mse_loss(net(x_curve), y_curve).backward()
        grad_first = net.first.weight.grad.norm().item()  # Größe des Gradienten der ersten Schicht
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        for step in range(300):
            loss = F.mse_loss(net(x_curve), y_curve)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        name = "mit Abkürzung " if residual else "ohne Abkürzung"
        print(f"{depth:2} Schichten {name}: Gradient erste Schicht {grad_first:9.1e}, "
              f"Verlust nach 300 Schritten {loss.item():.3f}")

# %% [markdown]
# Ohne Abkürzung wird der Gradient der ersten Schicht mit jeder Schicht kleiner: rund 0,05 bei
# 3 Schichten, rund 0,00006 bei 10 und rund 10 hoch minus 13 bei 30 Schichten. Er schrumpft also
# grob um den Faktor 3 pro Schicht. Das Netz mit 30 Schichten lernt deshalb gar nichts; sein
# Verlust bleibt bei 2,07, genau der Varianz der Zielwerte. Es sagt für jede Eingabe den
# Mittelwert voraus. Mit Abkürzung verschwindet der Gradient nicht, auch das Netz mit 30
# Schichten erreicht einen kleinen Verlust. Dass der Gradient mit der Tiefe sogar wächst, ist
# die umgekehrte Gefahr, das Explodieren; deshalb enthalten echte ResNets zusätzlich
# BatchNorm-Schichten, wie der Block unten.
#
# Warum hilft die Addition? Beim Rückwärtsrechnen wird der Gradient von Schicht zu Schicht
# weitergereicht. Ohne Abkürzung muss er durch jede Schicht hindurch und wird dabei jedes Mal mit
# deren Gewichten verrechnet; ist der Faktor kleiner als 1, schrumpft er wie ein Kapital, das
# jedes Jahr zwei Drittel seines Werts verliert. Bei der Addition gibt es zusätzlich einen
# direkten Weg, auf dem der Gradient unverändert durchgereicht wird. Die Schicht muss dann auch
# nicht mehr die ganze gewünschte Ausgabe lernen, sondern nur, was zur Eingabe noch hinzukommen
# soll. Dieser Unterschied heißt im Buch Residuum, daher der Name Residual Network.
#
# Das Buch zeigt auf S. 195 f. einen solchen Block mit zwei Faltungsschichten. Der Abdruck enthält
# ein paar Tippfehler: `input_channels` statt `in_channels`, und auch die zweite Faltung bekommt
# dort `stride`, was die Größe ein zweites Mal ändern würde. Hier eine lauffähige Fassung. Ändert
# die erste Faltung die Größe der Feature Map (`stride=2`) oder die Kanalzahl, passen Eingabe und
# Ausgabe nicht mehr zusammen und lassen sich nicht addieren. Für diesen Fall nennt das Buch eine
# Projektion der Eingabe auf die passende Form; in PyTorchs ResNet ist das eine Faltung mit
# Kerngröße 1 und demselben `stride`.

# %%
class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.projection = None
        if stride != 1 or in_channels != out_channels:
            # 1x1-Faltung bringt die Eingabe auf dieselbe Form wie die Ausgabe
            self.projection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        residual = x if self.projection is None else self.projection(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)


x = torch.randn(1, 32, 16, 16)
print(f"Eingabe:                          {tuple(x.shape)}")
print(f"Block 32 -> 32 Kanäle, stride 1:  {tuple(ResNetBlock(32, 32)(x).shape)}  (direkt addierbar)")
print(f"Block 32 -> 64 Kanäle, stride 2:  {tuple(ResNetBlock(32, 64, stride=2)(x).shape)}  (mit Projektion)")

# %% [markdown]
# Der erste Block behält Form und Kanalzahl, die Eingabe wird direkt addiert. Der zweite
# halbiert die Größe der Feature Map auf 8 × 8 und verdoppelt die Kanäle auf 64; hier bringt die
# 1×1-Faltung die Eingabe vor der Addition auf dieselbe Form.
#
# Das Buch stellt in Kapitel 8 zwei weitere Architekturen vor, die die Folien ebenfalls
# ansprechen. Ein Inception-Block (S. 202 ff.) wendet auf dieselbe Eingabe mehrere Faltungen mit
# verschiedenen Kerngrößen (1×1, 3×3, 5×5) und ein Pooling nebeneinander an und hängt die
# Ergebnisse entlang der Kanalachse aneinander (`torch.cat(outputs, 1)`). Das Netz muss sich
# also nicht vorab für eine Filtergröße entscheiden. DenseNet (S. 210 ff.) verbindet jede Schicht
# eines Blocks mit allen folgenden: Jede Schicht bekommt die aneinandergehängten Ausgaben aller
# vorherigen als Eingabe. ResNet addiert also, Inception und DenseNet hängen Kanäle aneinander.
# Für die Aufgabe brauchen Sie nur ResNet bzw. ResNeXt.

# %% [markdown]
# ## 3. Das ResNeXt im Encoder
#
# Nach dem einzelnen Block aus Abschnitt 2 sehen wir uns nun das ganze Netz an, das die Aufgabe
# verwendet. Den Aufbau kann man ohne Download ansehen, indem man `weights=None` übergibt; dann
# sind die Gewichte zufällig. Zum Vergleich zählen wir auch die Parameter von VGG16 aus Termin 2.

# %%
from torchvision.models import resnext101_32x8d, vgg16

resnext = resnext101_32x8d(weights=None)  # nur der Aufbau, keine Gewichte, kein Download


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


print(f"ResNeXt-101 (32x8d): {count_parameters(resnext) / 1e6:5.1f} Mio. Parameter")
print(f"VGG16:               {count_parameters(vgg16(weights=None)) / 1e6:5.1f} Mio. Parameter")
print()
print("Kinder von resnext, also list(resnext.children()):")
for name, child in resnext.named_children():
    print(f"  {name:8} {type(child).__name__}")
print(f"\nresnext.fc = {resnext.fc}")

# %% [markdown]
# ResNeXt-101 hat mit rund 89 Mio. Parametern deutlich weniger als VGG16 mit rund 138 Mio., obwohl
# es viel tiefer ist. Das Buch erklärt das auf S. 198: Die modernen Netze verzichten auf die großen
# vollverbundenen Schichten, in denen bei VGG16 der Großteil der Parameter steckt.
#
# Auf der obersten Ebene besteht das Netz aus zehn Teilen. `layer1` bis `layer4` enthalten die
# Residual-Blöcke, insgesamt 33. Danach fasst `avgpool` jede der 2048 Feature Maps zu einem
# einzigen Wert zusammen, und `fc` ordnet diese 2048 Werte den 1000 ImageNet-Klassen zu.
#
# `EncoderCNN` in der Aufgabe übernimmt mit `list(resnet.children())[:-1]` alle Teile außer dem
# letzten, also alles bis einschließlich `avgpool`. Das Buch macht auf S. 199 mit ResNet-34
# dasselbe. Übrig bleibt ein Netz, das zu jedem Bild 2048 Werte liefert statt 1000 Klassenwerte.
# Die Zahl 2048 liest der Code über `resnet.fc.in_features` ab, damit sie nicht von Hand
# eingetragen werden muss.

# %%
encoder_part = nn.Sequential(*list(resnext.children())[:-1])
photo = torch.randn(1, 3, 224, 224)  # ein Zufallsbild in der Größe der Aufgabe
with torch.no_grad():
    features = encoder_part(photo)
print(f"Ausgabe von children()[:-1]: {tuple(features.shape)} = 1 Bild, 2048 Kanäle, je 1 x 1")
print(f"nach features.view(features.size(0), -1): {tuple(features.view(features.size(0), -1).shape)}")
print(f"resnext.fc.in_features = {resnext.fc.in_features}")
print()
print("Mittlere Faltung im ersten Block von layer1:")
print("  ", resnext.layer1[0].conv2)

# %% [markdown]
# Nach `avgpool` hat die Ausgabe die Form (1, 2048, 1, 1): Jede der 2048 Feature Maps ist auf ein
# einziges Pixel zusammengefasst. `view(features.size(0), -1)` in `EncoderCNN.forward` macht
# daraus eine Zeile mit 2048 Werten je Bild.
#
# Was ResNeXt von ResNet unterscheidet, zeigt die letzte Zeile. Die mittlere Faltung jedes Blocks
# hat den Parameter `groups=32`: Die 256 Kanäle werden in 32 Gruppen zu je 8 Kanälen aufgeteilt,
# und jede Gruppe wird mit eigenen Filtern gefaltet, die nur die 8 Kanäle ihrer Gruppe sehen.
# Daher der Name `32x8d`. Das spart Parameter und lässt den Block mehrere schmale Wege
# nebeneinander lernen, ähnlich wie die parallelen Zweige in Inception. Die Abkürzungsverbindung
# ist dieselbe wie bei ResNet.

# %%
del resnext, encoder_part  # Speicher freigeben, das große Netz wird nicht mehr gebraucht

# %% [markdown]
# Für unsere Ersatzbilder ersetzen wir ResNeXt-101 durch ein kleines Netz mit drei der Blöcke aus
# Abschnitt 2. Die Teile heißen wie bei torchvision (`conv1`, `bn1`, `relu`, `layer1` bis
# `layer3`, `avgpool`, `fc`), damit `children()[:-1]` und `fc.in_features` genauso funktionieren.
#
# ResNeXt wurde auf ImageNet darauf trainiert, Bilder einer von 1000 Klassen zuzuordnen; das
# Bildbeschreiben hat es nie gelernt. Für den Ersatz bilden wir das nach: Wir trainieren das
# kleine Netz zuerst auf eine andere Aufgabe, nämlich Bilder mit nur einer Form einer der 12
# Kombinationen aus Farbe und Form zuzuordnen. Erst danach verwenden wir es als Encoder. Das ist
# Transfer Learning wie in Termin 2.

# %%
class MiniResNet(nn.Module):
    def __init__(self, num_classes=12):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.layer1 = ResNetBlock(32, 32)
        self.layer2 = ResNetBlock(32, 64, stride=2)
        self.layer3 = ResNetBlock(64, 128, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def make_single_object_images(n):
    images, labels = [], []
    for _ in range(n):
        color, shape = random_object()
        (place,) = random_places(1)
        images.append(draw([(color, shape, *place)]))
        labels.append(list(COLORS).index(color) * 3 + SHAPES.index(shape))  # Klasse 0 bis 11
    return torch.stack(images), torch.tensor(labels)


random.seed(1)
torch.manual_seed(1)
single_train_x, single_train_y = make_single_object_images(3000)
single_val_x, single_val_y = make_single_object_images(500)

backbone = MiniResNet()
optimizer = torch.optim.Adam(backbone.parameters(), lr=1e-3)
for epoch in range(3):
    backbone.train()
    order = torch.randperm(len(single_train_x))
    for start in range(0, len(order), 64):
        batch = order[start:start + 64]
        loss = F.cross_entropy(backbone(single_train_x[batch]), single_train_y[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    backbone.eval()
    with torch.no_grad():
        accuracy = (backbone(single_val_x).argmax(1) == single_val_y).float().mean().item()
    print(f"Vortraining Epoche {epoch + 1}: Genauigkeit auf 500 Validierungsbildern {accuracy:.1%}")
print(f"MiniResNet: {count_parameters(backbone) / 1e6:.2f} Mio. Parameter, fc.in_features = {backbone.fc.in_features}")

# %% [markdown]
# Nach drei Epochen ordnet das kleine Netz alle Validierungsbilder richtig zu. Es hat rund
# 0,3 Mio. Parameter, ein Dreihundertstel von ResNeXt-101, und liefert nach `avgpool` 128 statt
# 2048 Werte je Bild. Für das Bildbeschreiben übernehmen wir es wie die Aufgabe ResNeXt: ohne
# `fc` und ohne weiteres Training.

# %% [markdown]
# ## 4. Eingefroren, aber nicht ganz
#
# Das vortrainierte Netz aus Abschnitt 3 bekommt nun den Kopf, den `EncoderCNN` in der Aufgabe
# ergänzt: eine lineare Schicht `embed`, die die 2048 Werte (beim Ersatznetz 128) auf `embed_size`
# Werte abbildet, und eine `BatchNorm1d`, die diese Werte über den Batch normiert. Die Klasse
# unten ist die aus der Aufgabe; nur bekommt sie das vortrainierte Netz als Argument, statt
# ResNeXt selbst zu laden.

# %%
class EncoderCNN(nn.Module):
    def __init__(self, backbone, embed_size):
        super().__init__()
        modules = list(backbone.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(backbone.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.bn(features)
        return features


embed_size = 64
torch.manual_seed(2)
encoder = EncoderCNN(backbone, embed_size)
image_vectors = encoder(images)
print(f"encoder(images): {tuple(image_vectors.shape)} = 64 Bilder, je embed_size = {embed_size} Werte")

# %% [markdown]
# Das Buch friert ein vortrainiertes Netz ein, indem es bei allen Parametern
# `requires_grad = False` setzt (S. 199). `EncoderCNN` geht anders vor: Der Durchlauf durch
# `self.resnet` steht in `with torch.no_grad()`. Für diesen Teil merkt sich PyTorch dann keine
# Rechenschritte, und `backward()` kann keine Gradienten dorthin zurückrechnen. Die Gewichte
# bleiben also fest, auch wenn sie in der Liste `params` stehen, die die Aufgabe dem Optimierer
# übergibt; dort steht `encoder.resnet.parameters()` mit drin. Ein Trainingsschritt zeigt das.
# Ausgegeben wird außerdem, ob sich die BatchNorm-Schichten im eingefrorenen Teil verändern.

# %%
params = list(encoder.embed.parameters()) + list(encoder.bn.parameters()) + list(encoder.resnet.parameters())
step_optimizer = torch.optim.AdamW(params, lr=0.001)

frozen_weight = encoder.resnet[0].weight            # erste Faltung des eingefrorenen Teils
frozen_bn = encoder.resnet[1]                       # erste BatchNorm des eingefrorenen Teils
weight_before = frozen_weight.detach().clone()
running_mean_before = frozen_bn.running_mean.clone()

encoder.train()                                     # wie in Trainer.train()
loss = encoder(images).pow(2).mean()                # irgendein Verlust, nur für den Test
step_optimizer.zero_grad()
loss.backward()
step_optimizer.step()

print(f"requires_grad der ersten Faltung:        {frozen_weight.requires_grad}")
print(f"Gradient der ersten Faltung:             {frozen_weight.grad}")
print(f"Gradient von embed vorhanden:            {encoder.embed.weight.grad is not None}")
print(f"Gewichte der ersten Faltung verändert:   {not torch.equal(weight_before, frozen_weight)}")
print(f"running_mean der ersten BatchNorm verändert: "
      f"{not torch.equal(running_mean_before, frozen_bn.running_mean)}")

# %% [markdown]
# Die Parameter der ersten Faltung haben weiterhin `requires_grad=True`, bekommen aber keinen
# Gradienten (`None`); der Optimierer überspringt sie, und die Gewichte bleiben gleich. Der Kopf
# `embed` bekommt einen Gradienten und wird trainiert. Dass `encoder.resnet.parameters()` in
# `params` steht, schadet also nicht, bewirkt aber auch nichts.
#
# Die letzte Zeile zeigt, dass sich trotzdem etwas ändert. BatchNorm normiert mit dem Mittelwert
# und der Streuung des aktuellen Batches und führt im Trainingsmodus nebenher einen gleitenden
# Mittelwert mit, `running_mean` und `running_var`, der später im `eval()`-Modus verwendet wird.
# Das sind keine Gewichte, sondern mitgeführte Statistiken, und sie ändern sich auch innerhalb von
# `torch.no_grad()`. Weil `Trainer.train()` mit `self.encoder.train()` den ganzen Encoder in den
# Trainingsmodus schaltet, passen sich die BatchNorm-Statistiken des ResNeXt beim Training an die
# COCO-Bilder an, während seine Gewichte fest bleiben. Wer den eingefrorenen Teil ganz unverändert
# lassen will, schaltet nach `encoder.train()` noch `encoder.resnet.eval()`.
#
# Für die `BatchNorm1d` im Kopf setzt `EncoderCNN` `momentum=0.01`. Dieser Wert legt fest, wie
# schnell der gleitende Mittelwert nachzieht: Mit jedem Batch rückt er nur um ein Hundertstel des
# Abstands zum Mittelwert des aktuellen Batches vor, bei der Voreinstellung 0,1 um ein Zehntel.
# Außerdem braucht diese Schicht im Trainingsmodus mindestens zwei Bilder im Batch, weil sich aus
# einem einzigen Wert keine Streuung berechnen lässt. Wer den Encoder im Trainingsmodus mit einem
# einzelnen Bild aufruft, bekommt diese Fehlermeldung:

# %%
encoder.train()
try:
    encoder(images[:1])
except ValueError as error:
    print("ValueError:", error)

encoder.eval()
print(f"im eval()-Modus geht es: {tuple(encoder(images[:1]).shape)}")

# %% [markdown]
# Im `eval()`-Modus verwendet BatchNorm die mitgeführten Statistiken, und ein einzelnes Bild ist
# kein Problem. Deshalb schaltet `Trainer.evaluate()` mit `self.encoder.eval()` um, bevor es
# Bilder einzeln beschreibt, und auch in der letzten Zelle der Aufgabe muss der Encoder im
# `eval()`-Modus sein.
#
# Das Buch geht beim Einfrieren noch einen Schritt weiter (S. 199 ff.): Es schickt alle Bilder
# einmal durch das eingefrorene Netz, speichert die Ergebnisse und trainiert danach nur noch den
# kleinen Kopf auf diesen vorab berechneten Merkmalen. Das spart in jeder Epoche den teuren
# Durchlauf durch das große Netz. Die Aufgabe macht das nicht, und das hat einen Grund:
# `transform_train` schneidet mit `RandomCrop(224)` in jeder Epoche einen anderen Ausschnitt aus
# und spiegelt mit `RandomHorizontalFlip()` zufällig. Jedes Bild sieht also in jeder Epoche etwas
# anders aus, und die Merkmale müssten jedes Mal neu berechnet werden.

# %% [markdown]
# ## 5. Der Decoder im Training
#
# Der Encoder aus Abschnitt 4 liefert zu jedem Bild einen Vektor mit `embed_size` Werten. Daraus
# soll der Decoder eine Beschreibung machen. Das Buch beschreibt diese Aufteilung auf S. 223 ff.
# allgemein: Ein Encoder fasst die Eingabe, hier das Bild, in einem Vektor fester Länge zusammen,
# ein Decoder erzeugt daraus eine Folge, hier die Wörter. Diese Aufteilung braucht man immer
# dann, wenn Eingabe und Ausgabe verschiedene Form haben: Bild und Satz, oder zwei Sätze
# verschiedener Länge bei einer Übersetzung.
#
# Die folgende Klasse ist `DecoderRNN` aus der Aufgabe, unverändert mit den englischen
# Kommentaren. Sie besteht aus drei Schichten: `embed` übersetzt Wortnummern in Vektoren wie die
# Embeddings in Termin 3, `gru` ist eine GRU mit zwei Schichten, und `linear` macht aus jedem
# Zustand der GRU eine Bewertung für jedes Wort im Vokabular. `sample` und `sample_beam_search`
# erzeugen Beschreibungen; sie sind Thema der Abschnitte 7 und 8.
#
# Das Buch (S. 225) und die Folie "Aufgabe: Übersetzung von Bildern in Text" zeichnen den Decoder
# als LSTM, die Aufgabe verwendet eine GRU. Für die Idee des Encoder-Decoder-Netzes ist das
# gleichgültig: Beide lesen eine Folge Schritt für Schritt und tragen einen Zustand weiter. Wie in
# Termin 3 hat die GRU zwei Gates statt drei und kommt mit einem Zustand aus, während das LSTM
# zusätzlich sein Gedächtnis, den Cell State, führt; sie hat dadurch rund ein Viertel weniger
# Parameter. Wer `nn.GRU` durch `nn.LSTM` ersetzt, muss beachten, dass der Zustand dann ein Paar
# aus zwei Tensoren ist; `sample` und `sample_beam_search` reichen ihn aber ohnehin nur
# unverändert weiter.

# %%
from torch.nn import functional as F


class DecoderRNN(nn.Module):
    """Decoder RNN based on 2-layer GRU model."""
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=2, dropout=0.3):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        # Store length of the padded captions - we need it re-pad the packed output sequences 
        total_length = captions.size(1)
        
        reduced_lengths = lengths
        embeddings = self.embed(captions)

        inputs = torch.cat((features.unsqueeze(1), embeddings), 1)

        # pack padded input sequences
        inputs = torch.nn.utils.rnn.pack_padded_sequence(inputs, reduced_lengths.cpu(), batch_first=True, enforce_sorted=False)
        x, _ = self.gru(inputs)
        # pad sequences again using the stored length of the padded captions
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True, total_length=total_length)
        outputs = self.linear(x)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        """Accept a pre-processed image tensor (inputs) and return predicted 
        sentence (list of tensor ids of length max_len). This is the greedy
        search approach.
        """
        sampled_ids = []
        for i in range(max_len):
            hiddens, states = self.gru(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            # Get the index (in the vocabulary) of the most likely integer that
            # represents a word
            predicted = outputs.argmax(1)
            sampled_ids.append(predicted.item())
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_beam_search(self, inputs, states=None, max_len=20, beam_width=5):
        """Accept a pre-processed image tensor and return the top predicted 
        sentences. This is the beam search approach.
        """
        # Top word idx sequences and their corresponding inputs and states
        idx_sequences = [[[], 0.0, inputs, states]]
        for _ in range(max_len):
            # Store all the potential candidates at each step
            all_candidates = []
            # Predict the next word idx for each of the top sequences
            for idx_seq in idx_sequences:
                hiddens, states = self.gru(idx_seq[2], idx_seq[3])
                outputs = self.linear(hiddens.squeeze(1))
                # Transform outputs to log probabilities to avoid floating-point 
                # underflow caused by multiplying very small probabilities
                log_probs = F.log_softmax(outputs, -1)
                top_log_probs, top_idx = log_probs.topk(beam_width, 1)
                top_idx = top_idx.squeeze(0)
                # create a new set of top sentences for next round
                for i in range(beam_width):
                    next_idx_seq, log_prob = idx_seq[0][:], idx_seq[1]
                    next_idx_seq.append(top_idx[i].item())
                    log_prob += top_log_probs[0][i].item()
                    # Indexing 1-dimensional top_idx gives 0-dimensional tensors.
                    # We have to expand dimensions before embedding them
                    inputs = self.embed(top_idx[i].unsqueeze(0)).unsqueeze(0)
                    all_candidates.append([next_idx_seq, log_prob, inputs, states])
            # Keep only the top sequences according to their total log probability
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_width]
        return [idx_seq[0] for idx_seq in idx_sequences]


# %% [markdown]
# In `forward` passiert das Wesentliche in einer Zeile: `torch.cat((features.unsqueeze(1),
# embeddings), 1)` stellt den Bildvektor als ersten Schritt vor die eingebetteten Wörter der
# Beschreibung. Die GRU sieht also zuerst das Bild und danach die Beschreibung Wort für Wort. Die
# folgende Zelle verfolgt die Formen mit dem Batch aus Abschnitt 1. `hidden_size` ist die Größe
# des Zustands der GRU, in der Aufgabe 512, hier 128.

# %%
hidden_size = 128
torch.manual_seed(3)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
encoder.train()
decoder.train()

features = encoder(images)
embeddings = decoder.embed(captions)
inputs = torch.cat((features.unsqueeze(1), embeddings), 1)
outputs = decoder(features, captions, lengths)
print(f"features:                 {tuple(features.shape)} = 64 Bilder x embed_size")
print(f"features.unsqueeze(1):    {tuple(features.unsqueeze(1).shape)} = ein Schritt je Bild")
print(f"embeddings:               {tuple(embeddings.shape)} = 64 x {max_length + 2} Positionen x embed_size")
print(f"inputs nach torch.cat:    {tuple(inputs.shape)} = ein Schritt mehr")
print(f"outputs:                  {tuple(outputs.shape)} = 64 x {max_length + 2} Positionen x vocab_size")

# %% [markdown]
# Die Eingabe der GRU hat 12 Schritte, einen für das Bild und elf für die Nummernfolge. Die
# Ausgabe hat wieder nur 11 Positionen, so viele wie `captions`; dafür sorgt
# `pad_packed_sequence` mit `total_length=total_length`. An jeder Position stehen 19 Werte, einer
# für jedes Wort im Vokabular. Welches Wort an welcher Position vorhergesagt werden soll, zeigt
# die folgende Tabelle für die erste Beschreibung im Batch, die kürzer als die längste ist.

# %%
example = int((lengths < max_length + 2).nonzero()[0])  # erste Beschreibung mit Padding
example_length = lengths[example].item()
print(f"Beschreibung: {' '.join(idx2word[i] for i in captions[example][:example_length])}"
      f"   (length = {example_length})")
print(f"{'Position':>8}  {'Eingabe der GRU':<16}  {'soll vorhersagen (captions)':<26}")
for position in range(max_length + 2):
    given = "Bildvektor" if position == 0 else idx2word[captions[example][position - 1]]
    target = idx2word[captions[example][position]]
    note = "" if position < example_length else "  (nach length: nicht verwendet)"
    print(f"{position:>8}  {given:<16}  {target:<26}{note}")

# %% [markdown]
# An jeder Position soll das Netz das nächste Wort vorhersagen. An Position 0 sieht es nur das Bild
# und soll `<start>` liefern, an Position 1 sieht es zusätzlich `<start>` und soll das erste Wort
# liefern, und so weiter bis `<end>`. Als Eingabe bekommt es dabei immer die echte Beschreibung,
# nicht seine eigene Vorhersage vom Schritt davor. Dieses Vorgehen heißt Teacher Forcing: Wie ein
# Lehrer, der nach jedem Wort das richtige vorsagt, verhindert es, dass sich ein früher
# Fehler durch den ganzen Satz zieht. Erst beim Erzeugen in Abschnitt 7 muss das Netz mit seinen
# eigenen Wörtern weitermachen.
#
# `pack_padded_sequence` aus Termin 3 bekommt `lengths` und verarbeitet nur die ersten `length`
# Schritte jeder Folge. Da `length` die Nummern einschließlich `<start>` und `<end>` zählt und
# vorne der Bildvektor hinzukommt, endet die Eingabe genau beim letzten Wort, und die letzte
# verwendete Vorhersage ist `<end>`. Die Positionen dahinter füllt `pad_packed_sequence` mit
# Nullen auf.
#
# Das Buch beschreibt auf S. 225 eine Variante, bei der der Decoder den Bildvektor bei jedem
# Schritt erneut bekommt, weil es ihm sonst schwerfallen kann, sich über den ganzen Satz an das
# Bild zu erinnern. Die Aufgabe gibt ihn nur einmal am Anfang hinein; danach steckt die
# Information über das Bild allein im Zustand der GRU. Bei unseren kurzen Sätzen reicht das.
#
# Für den Verlust formt `Trainer.train()` die Ausgaben mit `outputs.view(-1, vocab_size)` zu einer
# langen Liste um, eine Zeile je Position, und `captions.view(-1)` entsprechend. Dann ist es eine
# gewöhnliche Klassifikation mit `vocab_size` Klassen und Kreuzentropie. Der `Trainer` erzeugt
# das Kriterium mit `nn.CrossEntropyLoss(ignore_index=pad_idx)`: Positionen, deren Zielwort
# `<pad>` ist, zählen nicht mit. Der Vergleich zeigt, wie viele Positionen das betrifft.

# %%
pad_idx = word2idx[PAD]
flat_outputs = outputs.view(-1, vocab_size)
flat_captions = captions.view(-1)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
print(f"Positionen im Batch:               {flat_captions.numel()}")
print(f"davon mit Zielwort <pad>:          {(flat_captions == pad_idx).sum().item()}")
print(f"Verlust mit ignore_index=pad_idx:  {criterion(flat_outputs, flat_captions).item():.3f}")
print(f"Verlust ohne ignore_index:         {nn.CrossEntropyLoss()(flat_outputs, flat_captions).item():.3f}")

# %% [markdown]
# Gut jede zwölfte Position im Batch hat `<pad>` als Ziel. Mit `ignore_index` wird der Verlust
# nur über die übrigen Positionen gemittelt. Dass beide Werte hier fast gleich sind, liegt am
# noch untrainierten Decoder: Er bewertet alle 19 Wörter ungefähr gleich, und das ergibt an jeder
# Position einen Verlust nahe dem natürlichen Logarithmus von 19, also 2,94, egal ob dort ein
# Wort oder `<pad>` stehen soll. Im Training macht `ignore_index` aber einen Unterschied. Ohne
# würde das Netz zusätzlich darauf trainiert, an diesen Positionen `<pad>` vorherzusagen. Das ist
# eine sinnlose Aufgabe, denn wegen `pack_padded_sequence` bekommt die GRU diese Positionen nie zu
# sehen; ihre Ausgabe dort ist immer dieselbe aufgefüllte Null. Bei COCO, wo die meisten
# Beschreibungen viel kürzer als die längste sind, wäre der größte Teil aller Positionen Padding
# und würde den Verlust beherrschen.

# %% [markdown]
# ## 6. Training des Ersatzmodells
#
# Mit den Bausteinen aus den Abschnitten 4 und 5 können wir trainieren. Die Schleife folgt
# `Trainer.train()`: Encoder und Decoder in den Trainingsmodus, für jeden Batch Bildvektoren
# berechnen, Decoder aufrufen, Ausgaben umformen, Verlust, `zero_grad`, `backward`, `step`. Die
# Parameterliste und der Optimierer `AdamW` sind wie in der Aufgabe gewählt, nur die Lernrate ist
# größer, damit drei Epochen reichen. Zusätzlich berechnen wir nach jeder Epoche den Verlust auf
# den Validierungsbildern. Das dauert etwa eine Minute.

# %%
from tqdm import tqdm  # Textbalken, kein Widget

torch.manual_seed(4)
encoder = EncoderCNN(backbone, embed_size)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
params = list(decoder.parameters()) + list(encoder.embed.parameters()) + list(encoder.bn.parameters()) + list(encoder.resnet.parameters())
optimizer = torch.optim.AdamW(params=params, lr=0.003, weight_decay=0.05, amsgrad=True)


def mean_loss(loader):
    encoder.eval()
    decoder.eval()
    total, batches = 0.0, 0
    with torch.no_grad():
        for images, captions, lengths, _ in loader:
            outputs = decoder(encoder(images), captions, lengths)
            total += criterion(outputs.view(-1, vocab_size), captions.view(-1)).item()
            batches += 1
    return total / batches


train_losses, val_losses = [], []
for epoch in range(3):
    encoder.train()
    decoder.train()
    total, batches = 0.0, 0
    for images, captions, lengths, _ in tqdm(train_loader, desc=f"Epoche {epoch + 1}"):
        outputs = decoder(encoder(images), captions, lengths)
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        batches += 1
    train_losses.append(total / batches)
    val_losses.append(mean_loss(val_loader))
    print(f"Epoche {epoch + 1}: Trainingsverlust {train_losses[-1]:.3f}, Validierungsverlust {val_losses[-1]:.3f}")

plt.figure(figsize=(5, 3))
plt.plot(range(1, 4), train_losses, marker="o", label="Training")
plt.plot(range(1, 4), val_losses, marker="o", label="Validierung")
plt.xlabel("Epoche")
plt.ylabel("Kreuzentropie je Wort")
plt.xticks([1, 2, 3])
plt.legend()
plt.show()

# %% [markdown]
# Der Verlust fällt in der ersten Epoche von 2,9 beim untrainierten Netz auf rund 0,3 und sinkt
# danach nur noch wenig. Der Trainingsverlust der ersten Epoche ist mit 0,58 höher, weil er über
# die ganze Epoche gemittelt ist, also auch über die ersten, noch schlechten Batches.
# Trainings- und Validierungsverlust liegen sonst eng beieinander; das Modell lernt also keine
# einzelnen Trainingsbilder auswendig. Null wird der Verlust nie, und das liegt an den Daten:
# Nach "a" kann jede der beiden Formen folgen, und nach der ersten Form jedes der vier
# Verbindungswörter. Wo die Beschreibungen selbst zufällig sind, kann das Netz nur die
# Wahrscheinlichkeiten lernen. Die Kreuzentropie ist an jeder Position der negative natürliche
# Logarithmus der Wahrscheinlichkeit, die das Netz dem richtigen Wort gibt, und der natürliche
# Logarithmus von 0,78 ist etwa minus 0,25. Ein Verlust von 0,25 je Wort entspricht also im Mittel
# einer Wahrscheinlichkeit von knapp 0,8 für das richtige nächste Wort.
#
# Der `Trainer` der Aufgabe speichert nach jeder Epoche den Trainingsstand, damit ein
# abgebrochenes Training weiterlaufen kann und das beste Modell erhalten bleibt. `save_as` legt
# dazu ein Wörterbuch mit `torch.save` ab. Die Einträge `encoder`, `decoder` und `optimizer` sind
# `state_dict()`s: Wörterbücher, die jedem Parameter und jeder mitgeführten Statistik ihren Namen
# und ihren Tensor zuordnen, beim Optimierer auch seine internen Werte. `load` lädt die Datei mit
# `torch.load` und schreibt die Tensoren mit `load_state_dict` in bereits erzeugte Modelle
# zurück. Das Modell muss also vorher mit denselben Größen erzeugt worden sein.

# %%
import os

torch.save({"encoder": encoder.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": 3}, "z06-checkpoint.pkl")

checkpoint = torch.load("z06-checkpoint.pkl")
print("Schlüssel im Checkpoint:", list(checkpoint))
print("Einige Einträge im state_dict des Decoders:")
for name, tensor in list(checkpoint["decoder"].items())[:4]:
    print(f"  {name:<20} {tuple(tensor.shape)}")

decoder_loaded = DecoderRNN(embed_size, hidden_size, vocab_size)  # neue, zufällige Gewichte
decoder_loaded.load_state_dict(checkpoint["decoder"])
print("Geladener Decoder hat dieselben Gewichte:",
      torch.equal(decoder_loaded.linear.weight, decoder.linear.weight))

os.remove("z06-checkpoint.pkl")  # die Datei wird im Weiteren nicht gebraucht

# %% [markdown]
# Die Namen im `state_dict` folgen den Attributen der Klasse: `embed.weight` hat die Form
# (vocab_size, embed_size), also einen Vektor je Wort. Nach `load_state_dict` hat der neu erzeugte
# Decoder dieselben Gewichte wie der trainierte. Würde man ihn etwa mit anderer `hidden_size`
# erzeugen, bräche `load_state_dict` mit einem `RuntimeError` ab, der für jeden betroffenen
# Parameter die Form im Checkpoint und die Form im Modell nennt (`size mismatch`).
#
# In der Aufgabe lädt `trainer.load()` die Datei `models/current-model.pkl`. Die Zeile
# `if not os.path.exists(trainer.current_state_file)` sorgt dafür, dass beim allerersten Lauf
# zuerst eine Epoche trainiert wird, damit es überhaupt etwas zu laden gibt.

# %% [markdown]
# ## 7. Wort für Wort erzeugen
#
# Beim Training aus Abschnitt 6 kannte der Decoder an jeder Position die richtigen Wörter davor.
# Für ein neues Bild gibt es diese Wörter nicht; der Decoder muss mit seinen eigenen weitermachen.
# `sample` macht das auf die einfachste Weise: Es gibt den Bildvektor in die GRU, nimmt das Wort
# mit der höchsten Bewertung, gibt dessen Embedding als nächste Eingabe hinein und wiederholt das
# `max_len`-mal. Weil in jedem Schritt das gerade beste Wort genommen wird, heißt das Verfahren
# greedy (gierig).
#
# Wir nehmen ein Validierungsbild und gehen die Schritte von `sample` einzeln durch. Zusätzlich
# rechnen wir mit `softmax` die Bewertungen in Wahrscheinlichkeiten um; `sample` braucht das
# nicht, weil `argmax` vorher und nachher dasselbe Wort liefert. Der Encoder muss im
# `eval()`-Modus sein (Abschnitt 4), und `unsqueeze(1)` macht aus dem Bildvektor wie in der
# Aufgabe einen einzelnen Schritt der Form (1, 1, embed_size).

# %%
encoder.eval()
decoder.eval()
image_index = 11
with torch.no_grad():
    image_vector = encoder(val_images[image_index:image_index + 1]).unsqueeze(1)
print(f"image_vector: {tuple(image_vector.shape)} = 1 Bild, 1 Schritt, embed_size")
print("Referenzen:", val_captions[image_index])

plt.figure(figsize=(1.6, 1.6))
plt.imshow(val_images[image_index].permute(1, 2, 0))
plt.axis("off")
plt.show()

with torch.no_grad():
    inputs, states = image_vector, None
    for step in range(12):
        hiddens, states = decoder.gru(inputs, states)
        outputs = decoder.linear(hiddens.squeeze(1))
        probs = F.softmax(outputs, dim=-1)[0]
        top = probs.topk(3)  # die drei größten Wahrscheinlichkeiten und ihre Wortnummern
        predicted = outputs.argmax(1)
        candidates = ", ".join(f"{idx2word[i]} {p:.2f}" for p, i in zip(top.values.tolist(), top.indices.tolist()))
        print(f"Schritt {step:2}: gewählt {idx2word[predicted.item()]:<9} beste drei: {candidates}")
        inputs = decoder.embed(predicted).unsqueeze(1)

# %% [markdown]
# Im ersten Schritt sagt das Netz mit Wahrscheinlichkeit 1,00 `<start>` voraus; das hat es in
# Abschnitt 5 an Position 0 gelernt. Danach wird es an drei Stellen unsicher. Für das erste Wort
# hält es "a" für etwas wahrscheinlicher als "there", 0,58 gegen 0,42, und nimmt deshalb "a".
# Welche der beiden Formen zuerst kommt, ist fast ein Münzwurf. Nach der ersten Form verteilt es
# die Wahrscheinlichkeit auf die vier Verbindungswörter und nimmt "with" mit 0,40. Die Wörter
# dazwischen sind dagegen sicher. Nach `<end>` geht die Schleife einfach weiter, denn `sample` hört
# erst nach `max_len` Schritten auf. Was nach `<end>` kommt, hat das Netz nie gelernt; im Training
# war dort nichts mehr vorherzusagen. Hier sagt es einfach weiter `<end>` voraus.
#
# `sample` liefert die Wortnummern als Liste, beginnend mit `<start>`. `clean_sentence` im
# `Trainer` macht daraus einen lesbaren Satz: Es überspringt `<start>` und hört bei `<end>` auf.
# Die folgende Funktion ist dieselbe Methode, nur ohne `self`.

# %%
def clean_sentence(word_idx_list):
    sentence = []
    for vocab_id in word_idx_list:
        word = idx2word[vocab_id]
        if word == END:
            break
        if word != START:
            sentence.append(word)
    return " ".join(sentence)


with torch.no_grad():
    greedy_ids = decoder.sample(image_vector)
print(f"sample liefert {len(greedy_ids)} Nummern: {greedy_ids}")
print(f"clean_sentence: '{clean_sentence(greedy_ids)}'")

# %% [markdown]
# Ob dieser Satz insgesamt der wahrscheinlichste ist, sagt das gierige Vorgehen nicht. Die
# Wahrscheinlichkeit eines ganzen Satzes ist das Produkt der Wahrscheinlichkeiten seiner Wörter,
# jeweils gegeben die Wörter davor. Bei langen Sätzen wird dieses Produkt sehr klein: 20 Wörter mit
# je 0,1 ergeben ein Hundertstel eines Trillionstels, und bei 50 Wörtern liegt das Ergebnis unter
# der kleinsten Zahl, die `float32` darstellen kann. Es wird zu 0. Man rechnet deshalb mit den
# Logarithmen der Wahrscheinlichkeiten und addiert sie, denn der Logarithmus eines Produkts ist
# die Summe der Logarithmen. `sample_beam_search` macht das mit `F.log_softmax`. Die
# Log-Wahrscheinlichkeiten sind negativ; je näher an 0, desto wahrscheinlicher.

# %%
print("Produkt von 50 Wahrscheinlichkeiten 0.1 in float32:", torch.tensor(0.1).pow(50).item())
print("Summe der 50 Logarithmen:                         ", (50 * torch.tensor(0.1).log()).item())


def caption_log_prob(image_vector, caption):
    # Summe der Log-Wahrscheinlichkeiten von <start>, allen Wörtern und <end>
    ids = [word2idx[START]] + [word2idx.get(w, word2idx[UNK]) for w in caption.split()] + [word2idx[END]]
    inputs, states, total = image_vector, None, 0.0
    with torch.no_grad():
        for idx in ids:
            hiddens, states = decoder.gru(inputs, states)
            log_probs = F.log_softmax(decoder.linear(hiddens.squeeze(1)), dim=-1)
            total += log_probs[0, idx].item()
            inputs = decoder.embed(torch.tensor([idx])).unsqueeze(1)
    return total


greedy_caption = clean_sentence(greedy_ids)
print(f"\nLog-Wahrscheinlichkeit von '{greedy_caption}': {caption_log_prob(image_vector, greedy_caption):.2f}")
# die Referenzen dieses Bildes und ein Satz mit dem häufigsten Muster, das unter ihnen zufällig
# fehlt
for caption in sorted(set(val_captions[image_index])) + ["there is a green triangle and a red triangle"]:
    print(f"Log-Wahrscheinlichkeit von '{caption}': {caption_log_prob(image_vector, caption):.2f}")

# %% [markdown]
# Die Funktion `caption_log_prob` bewertet einen beliebigen Satz mit dem trainierten Decoder.
# Der gierig erzeugte Satz kommt auf rund minus 2,1, das entspricht einer Wahrscheinlichkeit von
# etwa 0,12. Unter den fünf Referenzen dieses Bildes kommt das Muster "there is" zufällig nicht
# vor; der Satz in der letzten Zeile ist deshalb von Hand ergänzt. Er kommt auf rund minus 1,5,
# also etwa 0,23, und ist damit fast doppelt so wahrscheinlich. Hier zeigt sich, wozu die
# ungleiche Verteilung der Satzmuster aus Abschnitt 1 dient: "a" ist als erstes Wort
# wahrscheinlicher, weil vier der fünf Muster mit "a" beginnen, aber jedes dieser Muster für
# sich ist seltener als "there is a ... and a ...". Das gierige Vorgehen entscheidet sich beim
# ersten Wort und kann das später nicht mehr korrigieren.

# %% [markdown]
# ## 8. Beam Search
#
# Beam Search behebt das Problem aus Abschnitt 7, indem es nicht nur den einen besten Anfang
# weiterverfolgt, sondern die `beam_width` besten. Die Folie "Beam Search Heuristik" zeigt das
# für zwei Kandidaten. `sample_beam_search` geht in jedem Schritt so vor:
#
# 1. Für jeden der bisher behaltenen Kandidaten berechnet die GRU die Log-Wahrscheinlichkeiten des
#    nächsten Wortes.
# 2. `topk(beam_width, 1)` wählt je Kandidat die `beam_width` besten Fortsetzungen. Aus
#    `beam_width` Kandidaten werden so `beam_width` mal `beam_width` neue.
# 3. Jeder neue Kandidat bekommt als Bewertung die Summe der Log-Wahrscheinlichkeiten all seiner
#    Wörter.
# 4. Nur die `beam_width` Kandidaten mit der höchsten Summe werden behalten.
#
# Die folgende Zelle führt diese Schritte mit `beam_width = 2` für unser Bild aus und gibt nach
# jedem Schritt die behaltenen Kandidaten aus. Der Code entspricht `sample_beam_search`, nur mit
# Ausgaben und nach 11 Schritten beendet.

# %%
beam_width = 2
with torch.no_grad():
    beams = [[[], 0.0, image_vector, None]]  # wie in sample_beam_search: Wörter, Summe, Eingabe, Zustand
    for step in range(11):
        candidates = []
        for words, score, inputs, states in beams:
            hiddens, new_states = decoder.gru(inputs, states)
            log_probs = F.log_softmax(decoder.linear(hiddens.squeeze(1)), -1)
            top_log_probs, top_idx = log_probs.topk(beam_width, 1)
            for k in range(beam_width):
                idx = top_idx[0, k].item()
                candidates.append([words + [idx], score + top_log_probs[0, k].item(),
                                   decoder.embed(top_idx[0, k:k + 1]).unsqueeze(0), new_states])
        beams = sorted(candidates, key=lambda c: c[1], reverse=True)[:beam_width]
        kept = " | ".join(f"{' '.join(idx2word[i] for i in w)} ({s:.2f})" for w, s, _, _ in beams)
        print(f"Schritt {step:2}: {kept}")

# %% [markdown]
# In Schritt 0 gibt es nur einen Kandidaten, und dessen zwei beste Fortsetzungen sind `<start>`
# und ein unsinniges "is" mit minus 9,7; das fällt im nächsten Schritt sofort heraus. In Schritt 1
# behält Beam Search beide Anfänge, "a" mit minus 0,55 und "there" mit minus 0,87. In Schritt 2
# kostet "there is" fast nichts, während nach "a" die Unsicherheit über die erste Form hinzukommt;
# "there is" liegt nun vorn. Ab Schritt 4 sind beide behaltenen Kandidaten "there"-Sätze, die sich
# nur in der Reihenfolge der Formen unterscheiden. Am Ende steht "there is a green triangle and a
# red triangle" mit minus 1,49, der Satz, den das gierige Vorgehen in Abschnitt 7 verpasst hat.
#
# Die Methode selbst gibt nur die Wortnummern der `beam_width` behaltenen Kandidaten zurück, ohne
# ihre Bewertungen, als Liste von Listen und den besten zuerst. Jede innere Liste beginnt mit
# `<start>` und hat `max_len` Einträge.

# %%
with torch.no_grad():
    beam_results = decoder.sample_beam_search(image_vector)
print(f"sample_beam_search liefert {len(beam_results)} Listen mit je {len(beam_results[0])} Nummern")
for rank, ids in enumerate(beam_results[:3]):
    print(f"Platz {rank + 1}, roh:  {' '.join(idx2word[i] for i in ids)}")
print()
print(f"greedy:            '{greedy_caption}'  ({caption_log_prob(image_vector, greedy_caption):.2f})")
beam_caption = clean_sentence(beam_results[0])
print(f"Beam Search (5):   '{beam_caption}'  ({caption_log_prob(image_vector, beam_caption):.2f})")

# %% [markdown]
# Mit der Standardbreite 5 steht der gierige Satz auf Platz 3. Die rohen Listen zeigen außerdem,
# dass auch `sample_beam_search` nach `<end>` weiterläuft, bis `max_len` erreicht ist; das Netz
# füllt die Plätze mit weiteren `<end>`. Was nach `<end>` kommt, hat das Netz nie gelernt; im
# Training war dort nichts mehr vorherzusagen. `clean_sentence` schneidet es ab, für den fertigen
# Satz ist es also gleichgültig.
#
# Für die Reihenfolge der Kandidaten ist es das nicht ganz. Die Bewertung, nach der
# `sample_beam_search` sortiert, enthält auch die Log-Wahrscheinlichkeiten der Schritte nach
# `<end>`. Die sind zwar nahe 0, aber nicht genau 0, und ein kürzerer Satz hat mehr solcher
# Schritte als ein längerer. Zwei Sätze, die bis `<end>` fast gleich gut sind, können dadurch die
# Plätze tauschen. Üblich ist deshalb, einen Kandidaten nicht weiter zu verlängern, sobald er
# `<end>` erzeugt hat, und ihn mit seiner Bewertung bis `<end>` zur Seite zu legen. Die folgende
# Funktion tut das; ansonsten arbeitet sie wie `sample_beam_search`. Wir vergleichen beide
# Varianten und das gierige Vorgehen auf allen 300 Validierungsbildern. Als Maß dient
# `caption_log_prob` des gewählten Satzes.

# %%
END_IDX = word2idx[END]


def beam_search_until_end(decoder, inputs, max_len=20, beam_width=5):
    beams, finished = [[[], 0.0, inputs, None]], []
    for _ in range(max_len):
        candidates = []
        for words, score, inputs, states in beams:
            hiddens, new_states = decoder.gru(inputs, states)
            log_probs = F.log_softmax(decoder.linear(hiddens.squeeze(1)), -1)
            top_log_probs, top_idx = log_probs.topk(beam_width, 1)
            for k in range(beam_width):
                candidates.append([words + [top_idx[0, k].item()], score + top_log_probs[0, k].item(),
                                   decoder.embed(top_idx[0, k:k + 1]).unsqueeze(0), new_states])
        beams = []
        for candidate in sorted(candidates, key=lambda c: c[1], reverse=True)[:beam_width]:
            # wer <end> erreicht hat, wird zur Seite gelegt und nicht weiter verlängert
            (finished if candidate[0][-1] == END_IDX else beams).append(candidate)
        if not beams:
            break
    best_first = sorted(finished + beams, key=lambda c: c[1], reverse=True)
    return [words for words, _, _, _ in best_first]


# %%
results = {"greedy": [], "sample_beam_search": [], "beam_search_until_end": []}
with torch.no_grad():
    val_vectors = encoder(val_images).unsqueeze(1)  # (300, 1, embed_size)
    for i in tqdm(range(len(val_images)), desc="Validierungsbilder"):
        v = val_vectors[i:i + 1]
        results["greedy"].append(clean_sentence(decoder.sample(v)))
        results["sample_beam_search"].append(clean_sentence(decoder.sample_beam_search(v)[0]))
        results["beam_search_until_end"].append(clean_sentence(beam_search_until_end(decoder, v)[0]))

for name, sentences in results.items():
    mean_lp = sum(caption_log_prob(val_vectors[i:i + 1], s) for i, s in enumerate(sentences)) / len(sentences)
    with_there = sum(s.startswith("there") for s in sentences)
    print(f"{name:<22} mittlere Log-Wahrscheinlichkeit {mean_lp:6.2f}, beginnt mit 'there': {with_there:3} von 300")
differ = sum(a != b for a, b in zip(results["sample_beam_search"], results["beam_search_until_end"]))
print(f"\nsample_beam_search und beam_search_until_end wählen bei {differ} von 300 Bildern verschiedene Sätze")

# %% [markdown]
# Beide Beam-Search-Varianten finden im Mittel deutlich wahrscheinlichere Sätze als das gierige
# Vorgehen: rund minus 1,5 statt minus 1,95. Sie wählen fast immer das Muster mit "there", das
# gierig nur bei 58 von 300 Bildern herauskommt. Zwischen den beiden Beam-Search-Varianten ist der
# Unterschied klein, aber sichtbar: Bei etwa jedem zehnten Bild wählen sie verschiedene Sätze, und
# `beam_search_until_end` kommt im Mittel auf eine etwas höhere Log-Wahrscheinlichkeit. In der
# Aufgabe verwenden Sie `sample_beam_search` so, wie es vorgegeben ist.
#
# Größere `beam_width` verfolgt mehr Kandidaten und kostet entsprechend mehr Rechenzeit: Bei
# `beam_width = 5` und `max_len = 20` ruft `sample_beam_search` die GRU für jedes Bild knapp
# hundertmal auf, `sample` nur zwanzigmal. Mit `beam_width = 1` ist Beam Search dasselbe wie das
# gierige Vorgehen. Wie viel eine größere Breite bringt, können Sie mit der folgenden Liste
# ausprobieren.

# %%
beam_widths = [1, 2, 3, 5]  # Hier ändern

with torch.no_grad():
    for width in beam_widths:
        sentences = [clean_sentence(beam_search_until_end(decoder, val_vectors[i:i + 1], beam_width=width)[0])
                     for i in range(100)]
        mean_lp = sum(caption_log_prob(val_vectors[i:i + 1], s) for i, s in enumerate(sentences)) / len(sentences)
        print(f"beam_width = {width}: mittlere Log-Wahrscheinlichkeit auf 100 Bildern {mean_lp:6.2f}")

# %% [markdown]
# Schon zwei Kandidaten reichen hier, um die "there"-Sätze zu finden; mehr Breite ändert dann
# kaum noch etwas. Bei echten Fotos und einem Vokabular mit mehreren Tausend Wörtern gibt es viel
# mehr Stellen, an denen sich der Decoder zu früh festlegen kann; die Aufgabe verwendet die
# Voreinstellung `beam_width=5`.

# %% [markdown]
# ## 9. Auswertung
#
# Die Log-Wahrscheinlichkeit aus Abschnitt 8 misst, wie überzeugt das Modell von seinem eigenen
# Satz ist, aber nicht, ob der Satz das Bild richtig beschreibt. Dafür vergleicht man die
# erzeugten Sätze mit den Referenzen. Bei unseren Ersatzbildern geht das sehr direkt: Ein Satz
# ist inhaltlich richtig, wenn er genau die beiden Formen des Bildes nennt, gleich in welcher
# Reihenfolge. Als Untergrenze dient ein Modell, das das Bild gar nicht ansieht und immer den
# häufigsten Satz der Trainingsbeschreibungen ausgibt.

# %%
def named_objects(sentence):
    words = sentence.split()
    return {(words[k], words[k + 1]) for k in range(len(words) - 1)
            if words[k] in COLORS and words[k + 1] in SHAPES}


truth = [named_objects(captions[0]) for captions in val_captions]
most_common = Counter(c for captions in train_captions for c in captions).most_common(1)[0][0]
results["immer derselbe Satz"] = [most_common] * len(val_images)
print(f"häufigster Trainingssatz: '{most_common}'")
print()
for name, sentences in results.items():
    correct = sum(named_objects(s) == t for s, t in zip(sentences, truth))
    in_references = sum(s in refs for s, refs in zip(sentences, val_captions))
    print(f"{name:<22} beide Formen richtig: {correct / len(truth):6.1%}   "
          f"Satz wörtlich unter den 5 Referenzen: {in_references / len(truth):6.1%}")

# %% [markdown]
# Inhaltlich liegen alle drei Verfahren zwischen 95 und 98 %. Wo das Modell sich irrt, nennt es
# meist die richtigen Farben und Formen, ordnet sie aber falsch zu: Aus einem blauen Quadrat und
# einem grünen Dreieck wird ein blaues Dreieck und ein grünes Quadrat. Die Rangfolge ist eine
# andere als bei der Log-Wahrscheinlichkeit in Abschnitt 8: `beam_search_until_end` findet die
# wahrscheinlicheren Sätze, `sample_beam_search` macht etwas weniger Fehler, nämlich 7 statt 11
# von 300. Was das Modell für wahrscheinlich hält, ist nicht automatisch richtig. Der immer
# gleiche Satz trifft nur bei gut 2 % der Bilder, denn es gibt 66 verschiedene Paare aus zwei
# der zwölf Kombinationen von Farbe und Form.
#
# Die zweite Spalte zeigt, wie schlecht ein wörtlicher Vergleich als Maß taugt. Ob ein Satz
# zufällig unter den fünf Referenzen steht, hängt davon ab, welche Muster beim Erzeugen der
# Referenzen gezogen wurden. Die gierigen Sätze mit "a" treffen deshalb seltener als die Sätze
# mit "there", obwohl sie inhaltlich fast genauso oft richtig sind.
#
# Bei COCO lässt sich der Inhalt nicht so einfach prüfen, und ein wörtlicher Vergleich mit den
# fünf frei formulierten Referenzen träfe fast nie. Die Aufgabe bewertet deshalb mit CIDEr. Das
# Maß zerlegt den erzeugten Satz und jede Referenz in Wortgruppen aus einem bis vier
# aufeinanderfolgenden Wörtern und misst, wie stark diese übereinstimmen. Wortgruppen, die in den
# Beschreibungen fast aller Bilder vorkommen, wie "a picture of", zählen dabei wenig; Wortgruppen,
# die für das Bild typisch sind, wie "riding a wave", zählen viel. Höhere Werte sind besser.
# `Trainer.evaluate()` berechnet CIDEr nach jeder Epoche und speichert das Modell mit dem bisher
# besten Wert als `best-model.pkl`.
#
# `COCOEvalCap` zerlegt die Sätze mit einem Java-Programm; ist auf dem Rechner kein Java
# installiert, bricht `evaluate()` mit `FileNotFoundError: ... 'java'` ab. Und `evaluate()`
# erzeugt für jedes der rund 40.000 Validierungsbilder einzeln eine Beschreibung mit
# `sample_beam_search`; das dauert entsprechend lange.

# %% [markdown]
# ## 10. Zusammenfassung
#
# - Das Vokabular beginnt mit `<pad>`, `<start>`, `<end>` und `<unk>`; Wörter, die seltener als
#   `vocab_threshold`-mal vorkommen, werden zu `<unk>`. Jede Beschreibung wird zu einer
#   Nummernfolge mit `<start>` und `<end>`, aufgefüllt mit `<pad>`, plus ihrer Länge.
# - Sehr tiefe Netze lernen schlecht, weil der Gradient auf dem Weg zurück zu den ersten Schichten
#   verschwindet. Die Abkürzungsverbindung von ResNet addiert die Eingabe eines Blocks auf seine
#   Ausgabe und gibt dem Gradienten einen direkten Weg; bei geänderter Form bringt eine
#   1×1-Faltung die Eingabe auf die passende Form. Inception und DenseNet hängen Kanäle aneinander,
#   statt zu addieren.
# - ResNeXt-101 hat weniger Parameter als VGG16 und teilt die mittlere Faltung jedes Blocks in 32
#   Gruppen. `children()[:-1]` entfernt die Klassifikationsschicht und lässt 2048 Werte je Bild
#   übrig.
# - `torch.no_grad()` im Encoder hält die Gewichte des ResNeXt fest, die BatchNorm-Statistiken
#   ändern sich im Trainingsmodus trotzdem. Die `BatchNorm1d` im Kopf braucht im Trainingsmodus
#   mindestens zwei Bilder; zum Beschreiben einzelner Bilder gehört der Encoder in `eval()`.
# - Der Decoder bekommt den Bildvektor als ersten Schritt und danach die echte Beschreibung
#   (Teacher Forcing); an jeder Position soll er das nächste Wort vorhersagen.
#   `ignore_index=pad_idx` nimmt das Padding aus dem Verlust.
# - Ein Checkpoint speichert die `state_dict()`s von Encoder, Decoder und Optimierer;
#   `load_state_dict` schreibt sie in Modelle gleicher Größe zurück.
# - `sample` wählt in jedem Schritt das beste Wort und kann dabei einen insgesamt
#   wahrscheinlicheren Satz verpassen. `clean_sentence` entfernt `<start>` und schneidet bei
#   `<end>` ab.
# - `sample_beam_search` verfolgt die `beam_width` besten Kandidaten nach der Summe ihrer
#   Log-Wahrscheinlichkeiten und gibt deren Wortnummern als Liste von Listen zurück, den besten
#   zuerst. Es läuft nach `<end>` weiter, was die Reihenfolge der Kandidaten verändern kann.
# - Die Aufgabe verwendet als Decoder eine GRU, Buch und Folien zeichnen ein LSTM; am Prinzip
#   ändert das nichts.
# - Ob ein Satz das Bild richtig beschreibt, misst die Log-Wahrscheinlichkeit nicht. Die Aufgabe
#   vergleicht mit den Referenzen über CIDEr, das übereinstimmende Wortgruppen zählt und
#   typische Wortgruppen stärker gewichtet.
