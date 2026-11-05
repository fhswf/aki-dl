# %% [markdown]
# # Ergänzung zu Termin 5: Style Transfer und DCGAN
#
# Beim Style Transfer in Aufgabe 1 wird kein Netz trainiert: Ein fertiges VGG19 bewertet, wie
# sehr ein Bild inhaltlich dem Foto der Fachhochschule und stilistisch der Sternennacht
# ähnelt, und der Optimierer verändert direkt die Pixel. In der Aufgabe verändern Sie
# `style_layers`, `content_layers` und die Gewichte; dafür muss man wissen, welche Schicht
# hinter einer Nummer steckt und was die Gram-Matrix misst. In Aufgabe 2 werden ein Generator
# und ein Diskriminator gegeneinander trainiert, zusammen ein Generative Adversarial Network
# (GAN). Die Variante DCGAN (Deep Convolutional GAN) baut beide Netze aus Faltungsschichten.
# In der Aufgabe bauen Sie die Netze nach einer Beschreibung und wählen die Verlustfunktion.
# Kapitel 7 des Buchs behandelt beide Verfahren, und die Aufgaben übernehmen weitgehend seinen
# Code.
#
# ```
# Aufgabe 1: Style Transfer
#   Bild ──► VGG19.features ──► Ausgaben einzelner Schichten     Abschnitte 1 und 2
#                  ├─► späte Schicht: Inhalt (MSE)
#                  └─► mehrere Schichten: Stil (Gram-Matrix)     Abschnitt 3
#   Optimierer ändert die Pixel des Bildes, nicht das Netz       Abschnitt 4
#   Layer und Gewichte variieren                                 Abschnitt 5
#
# Aufgabe 2: DCGAN
#   Zufallsvektor ──► Generator ──► Bild ──► Diskriminator       Abschnitt 6
#   BatchNorm, Initialisierung, LeakyReLU                        Abschnitt 7
#   Verlustfunktion                                              Abschnitt 8
#   Trainingsschleife: erst Diskriminator, dann Generator        Abschnitt 9
#   Training und Deutung der Kurven                              Abschnitte 10 und 11
# ```
#
# Teil A und Teil B sind unabhängig voneinander. Nach einem Neustart des Kernels führen Sie
# die Setup-Zelle und anschließend den gewünschten Teil von seinem Anfang an aus.

# %% [markdown]
# ## 0. Setup

# %%
try:
    import torch  # torch ist auf Colab und dem Jupyter-Server meist schon installiert
    import torchvision
    print(f"PyTorch {torch.__version__} und torchvision {torchvision.__version__} bereits vorhanden.")
except ImportError:
    !pip install -q torch torchvision

# matplotlib für die Bilder und Diagramme, tqdm für den Fortschrittsbalken beim Training
!pip install -q matplotlib==3.11.2 tqdm==4.70.1

# %% [markdown]
# # Teil A: Style Transfer
#
# ## 1. Was hinter den Layer-Nummern steckt
#
# Aufgabe 1 wählt Schichten über ihre Position in `vgg19(...).features` aus: `style_layers =
# [1, 6, 11, 20, 25]` und `content_layers = [-1, -5, -15]`. VGG19 ist aufgebaut wie VGG16 aus
# Termin 2, mit drei Faltungsschichten mehr. `features` ist der Faltungsteil des Netzes, also
# alle Faltungen mit ihren ReLUs und MaxPools, ohne die vollverbundenen Schichten am Ende. Um
# nur den Aufbau anzusehen, genügt `weights=None`; dann wird nichts heruntergeladen.

# %%
import torch.nn as nn
from torchvision.models import vgg19

vgg_layers = vgg19(weights=None).features  # nur der Aufbau, keine Gewichte, kein Download
print(f"vgg19().features hat {len(vgg_layers)} Schichten, Nummern 0 bis {len(vgg_layers) - 1}")


def describe(layer_number):
    layer = vgg_layers[layer_number]
    # Kanalzahl: bei Conv2d direkt ablesbar, bei ReLU und MaxPool die der letzten Conv2d davor
    position = layer_number % len(vgg_layers)  # -1 wird zu 36, -5 zu 32 usw.
    channels = next(vgg_layers[k].out_channels for k in range(position, -1, -1)
                    if isinstance(vgg_layers[k], nn.Conv2d))
    return f"Nummer {layer_number:>3} = Position {position:>2}: {type(layer).__name__:<9} ({channels} Kanäle)"


print("style_layers der Aufgabe:")
for number in [1, 6, 11, 20, 25]:
    print("  ", describe(number))
print("content_layers der Aufgabe:")
for number in [-1, -5, -15]:
    print("  ", describe(number))

# %% [markdown]
# Die Stil-Schichten 1, 6, 11 und 20 sind jeweils die ReLU nach der ersten Faltung der Blöcke
# mit 64, 128, 256 und 512 Kanälen, Nummer 25 ist eine Faltung im vierten Block. Negative
# Nummern zählen wie bei Python-Listen vom Ende.
#
# Das `n` in `style_weights = [1e3/n**2 for n in [...]]` ist die Kanalzahl der jeweiligen
# Stil-Schicht, im Buch `[64, 128, 256, 512, 512]`. Wählen Sie andere Schichten, passen Sie
# diese Zahlenliste entsprechend an.

# %% [markdown]
# ## 2. Hooks: Ausgaben einzelner Schichten mitschneiden
#
# Um an die Ausgaben der gewählten Schichten heranzukommen, verwendet die Aufgabe die Klasse
# `LayerActivations` und die Funktion `extract_layers`, beide aus dem Buch übernommen. Sie
# arbeiten mit Hooks. Ein Hook ist eine Funktion, die man bei einer Schicht anmeldet und die
# PyTorch dann bei jedem Durchlauf dieser Schicht mit deren Ausgabe aufruft.
#
# Statt mit den rund 550 MB großen Gewichten von VGG19 rechnen wir hier mit einem kleinen
# Ersatznetz. Es ist nach demselben Prinzip aus Faltung, ReLU und MaxPool aufgebaut, hat aber
# weniger Kanäle und zufällige statt trainierter Gewichte. Um zu sehen, wie Hooks, Gram-Matrix
# und Optimierung zusammenwirken, genügt das; mit einem trainierten VGG19 werden die Bilder
# besser.

# %%
torch.manual_seed(0)  # feste Zufallsgewichte bei jedem Lauf


def vgg_block(in_channels, out_channels):
    return [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True)]


small_vgg = nn.Sequential(*vgg_block(3, 32), *vgg_block(32, 32), nn.MaxPool2d(2),
                          *vgg_block(32, 64), *vgg_block(64, 64), nn.MaxPool2d(2),
                          *vgg_block(64, 128), *vgg_block(128, 128))
for module in small_vgg:
    if isinstance(module, nn.Conv2d):
        # übliche Startbelegung für ReLU-Netze: hält die Werte über die Schichten etwa gleich groß
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        nn.init.zeros_(module.bias)
for param in small_vgg.parameters():
    param.requires_grad = False  # wie in der Aufgabe: das Netz wird nicht trainiert
print(small_vgg)


# %% [markdown]
# Die Nummern 1, 6 und 11 sind wie bei VGG19 die ReLUs nach der ersten Faltung eines Blocks.
# `LayerActivations` und `extract_layers` sind unverändert aus der Aufgabe übernommen:

# %%
class LayerActivations():
    features = []  # Klassenattribut: gehört allen Objekten dieser Klasse gemeinsam

    def __init__(self, model, layer_nums):
        self.hooks = []
        for layer_num in layer_nums:
            # hook_fn wird künftig nach jedem Durchlauf dieser Schicht aufgerufen
            self.hooks.append(model[layer_num].register_forward_hook(self.hook_fn))

    def hook_fn(self, module, input, output):
        self.features.append(output)  # Ausgabe der Schicht merken

    def remove(self):
        for hook in self.hooks:
            hook.remove()  # Hook abmelden, sonst sammelt er bei jedem Durchlauf weiter


def extract_layers(layers, img, model=None):
    la = LayerActivations(model, layers)
    la.features = []  # neue, eigene Liste für genau diesen Aufruf
    out = model(img)  # der Durchlauf löst die Hooks aus; die Ausgabe selbst wird nicht gebraucht
    la.remove()
    return la.features


x = torch.rand(1, 3, 64, 64)  # ein Bild: (Batch, Kanäle, Höhe, Breite)
features = extract_layers([1, 6, 11], x, model=small_vgg)
for number, feature in zip([1, 6, 11], features):
    print(f"Schicht {number:>2}: {tuple(feature.shape)} = 1 Bild, {feature.shape[1]} Kanäle, "
          f"{feature.shape[2]} × {feature.shape[3]} Positionen")

# %% [markdown]
# Mit jedem MaxPool halbiert sich die Bildgröße, die Zahl der Kanäle steigt. `la.features =
# []` ist nötig, weil `features` ein Klassenattribut ist, das sich alle Objekte teilen; sonst
# wüchse die Liste bei jedem Aufruf weiter.

# %% [markdown]
# ## 3. Die Gram-Matrix
#
# Was macht die Aufgabe mit den Ausgaben, die `extract_layers` liefert? Für den Inhaltsverlust
# (Content Loss) vergleicht sie die Ausgaben einer späten Schicht für das neue Bild und für
# das Inhaltsbild direkt mit `nn.MSELoss`. Für den Stil wäre das zu streng: Das
# neue Bild soll Wirbel wie die Sternennacht haben, aber nicht an denselben Stellen. Für den
# Stil berechnet die Aufgabe deshalb die Gram-Matrix. Sie misst die Korrelation zwischen den
# Feature-Maps einer Schicht, also welche Kanäle gemeinsam aktiv sind. Ein kleines Beispiel wie im Buch, mit einem Bild aus 3 Kanälen zu je 3 × 3 Werten:

# %%
feature_map = torch.tensor([[[1., 0., 0.], [1., 0., 0.], [1., 0., 0.]],   # Kanal 0: linke Spalte
                            [[1., 0., 0.], [1., 0., 0.], [0., 0., 0.]],   # Kanal 1: oben und Mitte links
                            [[0., 0., 2.], [0., 0., 2.], [0., 0., 2.]]])  # Kanal 2: rechte Spalte
feature_map = feature_map.unsqueeze(0)  # (1, 3, 3, 3) = 1 Bild, 3 Kanäle, 3 × 3


class GramMatrix(nn.Module):  # wie in der Aufgabe
    def forward(self, input):
        b, c, h, w = input.size()  # Batch, Kanäle, Höhe, Breite
        features = input.view(b, c, h * w)  # je Kanal alle h * w Positionen in einer Zeile
        # Matrix mal ihre Transponierte, je Bild (bmm): jede Kanalzeile mit jeder -> (b, c, c)
        gram_matrix = torch.bmm(features, features.transpose(1, 2))
        gram_matrix.div_(h * w)  # durch die Zahl der Positionen teilen
        return gram_matrix


print("Je Kanal eine Zeile mit 9 Werten:")
print(feature_map.view(1, 3, 9)[0])  # [0]: das erste (und einzige) Bild
print("Gram-Matrix (Kanal × Kanal):")
print(GramMatrix()(feature_map)[0])

# %% [markdown]
# Die Gram-Matrix hat eine Zeile und eine Spalte je Kanal. Ein Eintrag entsteht, indem man die
# Zeilen zweier Kanäle Position für Position multipliziert, aufsummiert und durch die Zahl der
# Positionen, hier 9, teilt:
#
# 1. Kanal 0 und Kanal 1 sind an zwei Positionen gleichzeitig aktiv (oben links und Mitte
#    links). Das ergibt 2 geteilt durch 9, etwa 0,22.
# 2. Kanal 0 und Kanal 2 sind nie gleichzeitig aktiv, der Eintrag ist 0.
# 3. Auf der Diagonalen steht, wie stark ein Kanal insgesamt ist: Kanal 2 hat dreimal den Wert
#    2, also 3 mal 4 geteilt durch 9, etwa 1,33.
#
# Beim Aufsummieren geht verloren, wo im Bild etwas geschieht. Die Gram-Matrix gleicht damit
# einer Tabelle, die festhält, welche Sensoren einer Anlage gleichzeitig anschlagen, aber
# nicht wann.
#
# Was das bewirkt, zeigt ein Streifenbild, das um einige Pixel verschoben wird. Wir
# vergleichen die verschobenen Streifen und ein ganz anderes Bild mit dem Original, einmal
# direkt und einmal über die Gram-Matrix, jeweils an Schicht 6 des Ersatznetzes:

# %%
size = 64
# Koordinatengitter: für jedes Pixel seine Zeile (ys) und Spalte (xs)
ys, xs = torch.meshgrid(torch.arange(size * 1.0), torch.arange(size * 1.0), indexing="ij")

# Stilbild: diagonale Streifen in Rot, Blau und Gelb, je 6 Pixel breit
stripe_colors = torch.tensor([[0.85, 0.2, 0.15], [0.15, 0.3, 0.8], [0.95, 0.85, 0.2]])
stripe_index = ((xs + ys) / 6).floor().long() % 3  # welche der drei Farben an jedem Pixel
# Farbe je Pixel nachschlagen -> (64, 64, 3), dann umordnen zu (1, 3, 64, 64)
style_image = stripe_colors[stripe_index].permute(2, 0, 1).unsqueeze(0)

# Inhaltsbild: grauer Hintergrund, helle Scheibe, dunkles Quadrat
content_image = torch.full((1, 3, size, size), 0.5)  # überall Grauwert 0,5
content_image[0, :, ((xs - 22) ** 2 + (ys - 30) ** 2).sqrt() < 12] = 0.95
content_image[0, :, ((xs - 46).abs() < 9) & ((ys - 38).abs() < 9)] = 0.15

shifted_image = torch.roll(style_image, shifts=9, dims=3)  # Streifen um 9 Pixel nach rechts

import matplotlib.pyplot as plt

mse = nn.MSELoss()  # mittlerer quadratischer Fehler, wie im Inhaltsverlust der Aufgabe
with torch.no_grad():  # nur vergleichen, keine Gradienten nötig
    # [0]: extract_layers liefert eine Liste, hier mit nur einem Eintrag (Schicht 6)
    stripes, shifted, other = (extract_layers([6], image, model=small_vgg)[0]
                               for image in [style_image, shifted_image, content_image])
    print("                                   direkt (MSE)   Gram-Matrix (MSE)")
    print(f"Streifen vs. verschobene Streifen   {mse(stripes, shifted).item():10.4f}"
          f"   {mse(GramMatrix()(stripes), GramMatrix()(shifted)).item():12.6f}")
    print(f"Streifen vs. Scheibe und Quadrat    {mse(stripes, other).item():10.4f}"
          f"   {mse(GramMatrix()(stripes), GramMatrix()(other)).item():12.6f}")

fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.7))
for ax, image, title in zip(axes, [style_image, shifted_image, content_image],
                            ["Streifen", "verschoben", "Scheibe und Quadrat"]):
    ax.imshow(image[0].permute(1, 2, 0))  # imshow erwartet (Höhe, Breite, Kanäle)
    ax.set_title(title)
    ax.axis("off")
plt.show()


# %% [markdown]
# Direkt verglichen sind die verschobenen Streifen fast so weit vom Original entfernt wie das
# völlig andere Bild (etwa 0,15 gegen 0,17), denn an fast jeder Position sieht die Schicht
# jetzt etwas anderes als vorher. Über die Gram-Matrix ist der Abstand zu den verschobenen
# Streifen rund hundertmal kleiner als der zum anderen Bild. Genau so trennt die Aufgabe
# Inhalt und Stil: Der Inhaltsverlust achtet darauf, was wo im Bild ist, der Stilverlust
# (Style Loss) in `StyleLoss` nur darauf, welche Muster zusammen vorkommen. Weil frühe
# Schichten kleine und späte Schichten größere Muster erfassen, wird der Stil über mehrere
# Schichten gemessen.

# %%
class StyleLoss(nn.Module):  # wie in der Aufgabe

    def forward(self, inputs, targets):
        out = nn.MSELoss()(GramMatrix()(inputs), targets)  # targets ist schon eine Gram-Matrix
        return (out)


# Probe mit den Streifen von oben: derselbe Wert wie in der Tabelle, Spalte Gram-Matrix
print(f"StyleLoss(verschobene Streifen, Gram-Matrix der Streifen): "
      f"{StyleLoss()(shifted, GramMatrix()(stripes)).item():.6f}")

# %% [markdown]
# ## 4. Optimiert wird das Bild, nicht das Netz
#
# Das neue Bild entsteht, indem der Optimierer die Pixel so lange verändert, bis Inhalts- und
# Stilverlust klein sind; das Netz bleibt dabei fest. In der Aufgabe stecken dafür drei
# Zeilen:
#
# 1. `param.requires_grad = False` für alle Gewichte des Netzes.
# 2. `output_image = Variable(content_image.data.clone(), requires_grad=True)`: eine Kopie des
#    Inhaltsbildes, für die PyTorch Gradienten berechnet. Heute schreibt man dafür
#    `content_image.clone().requires_grad_(True)`.
# 3. `optim.LBFGS([output_image])`: Der Optimierer bekommt nur das Bild.
#
# Die folgende Zelle berechnet für eine solche Kopie einmal den Stilverlust gegenüber dem
# Streifenbild und ruft `backward()` auf:

# %%
output_image = content_image.clone().requires_grad_(True)  # Kopie, für die Gradienten berechnet werden
loss = StyleLoss()(extract_layers([6], output_image, model=small_vgg)[0],
                   GramMatrix()(extract_layers([6], style_image, model=small_vgg)[0]))
loss.backward()  # Gradienten berechnen, wie im Training

print(f"output_image.grad: {tuple(output_image.grad.shape)} = je Pixel und Farbkanal ein Gradient")
print(f"Gradient am ersten Gewicht des Netzes: {small_vgg[0].weight.grad}")

# %% [markdown]
# `backward()` legt den Gradienten in `.grad` ab. Er hat dieselbe Form wie das Bild und gibt
# für jeden Pixelwert an, wie sich der Verlust ändert, wenn man diesen Wert ein wenig erhöht;
# der Optimierer verschiebt die Pixel in die Gegenrichtung. Die Pixel sind hier also die
# Parameter. Die Gewichte des Netzes bekommen keinen Gradienten (`None`). Die Zielwerte macht
# die Aufgabe mit `detach()` zu festen Vergleichswerten ohne Gradient; verändert wird nur
# `output_image`.
#
# Bisher haben wir mit Adam optimiert, der gleitende Mittelwerte der Gradienten bildet. L-BFGS
# schätzt dagegen aus den letzten Schritten, wie stark die Verlustfläche gekrümmt ist; das
# meint die Aufgabe mit der Näherung der Hesse-Matrix. Er wertet den Verlust in jedem Schritt
# mehrmals aus und bekommt deshalb mit `closure` eine Funktion, die den Verlust jedes Mal neu
# berechnet. Der Fortschrittsbalken der Aufgabe zählt diese Auswertungen;
# `maximum_iterations=1000` sind also 1000 Auswertungen, nicht 1000 Schritte.
#
# Die folgende Funktion ist `style_transfer` aus der Aufgabe, nur mit Schichten und Gewichten
# als Argumenten und mit dem Bild als Rückgabe.

# %%
from functools import reduce
from tqdm import tqdm  # Textbalken, braucht kein Widget-JavaScript


def style_transfer(style_image, content_image, style_layers, content_layers,
                   style_weights, content_weights, start_image=None,
                   maximum_iterations=500, model=small_vgg):
    loss_layers = style_layers + content_layers
    weights = style_weights + content_weights

    # Ziele einmal berechnen; detach: feste Vergleichswerte ohne Gradient
    content_targets = [t.detach() for t in extract_layers(content_layers, content_image, model=model)]
    style_targets = [GramMatrix()(t).detach() for t in extract_layers(style_layers, style_image, model=model)]
    targets = style_targets + content_targets

    loss_fns = [StyleLoss()] * len(style_layers) + [nn.MSELoss()] * len(content_layers)

    # Startbild: Kopie des Inhaltsbildes wie in der Aufgabe, oder ein anderes
    start = content_image if start_image is None else start_image
    output_image = start.clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS([output_image])

    with tqdm(total=maximum_iterations) as pbar:
        while pbar.n < pbar.total:
            def closure():
                optimizer.zero_grad()
                out = extract_layers(loss_layers, output_image, model=model)
                layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
                loss = reduce(torch.add, layer_losses)  # gewichtete Summe aller Teilverluste
                loss.backward()
                pbar.update()
                return loss

            optimizer.step(closure)
    return output_image.detach().clamp(0, 1)  # wie postprocess_b: Werte auf 0 bis 1 begrenzen


def show_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(2.4 * len(images), 2.7))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image[0].permute(1, 2, 0))
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.show()


# %% [markdown]
# Als Stil-Schichten nehmen wir alle fünf ReLUs des Ersatznetzes, die Gewichte nach derselben
# Regel wie in der Aufgabe. Der erste Lauf kommt ohne Inhalt aus: kein Inhaltsverlust und
# reines Rauschen als Startbild. Übrig bleibt die Forderung, dass die Gram-Matrizen zu denen
# der Streifen passen.

# %%
style_layers = [1, 3, 6, 8, 11]
style_weights = [1e3 / n ** 2 for n in [32, 32, 64, 64, 128]]  # Kanalzahlen der Schichten

torch.manual_seed(1)
noise_image = torch.rand(1, 3, size, size)  # Startbild: Zufallsrauschen
texture = style_transfer(style_image, content_image, style_layers, content_layers=[-1],
                         style_weights=style_weights, content_weights=[0.0],  # kein Inhalt
                         start_image=noise_image)
show_images([style_image, noise_image, texture], ["Stilbild", "Startbild: Rauschen", "Ergebnis"])

# %% [markdown]
# Aus dem Rauschen sind diagonale Streifen in den drei Farben entstanden, aber an anderen
# Stellen als im Stilbild: Die Gram-Matrizen verlangen nur, dass die Muster vorkommen, nicht wo.
# Im zweiten Lauf kommt der Inhalt hinzu: Startbild ist wie in der Aufgabe das Inhaltsbild, und
# ein Inhaltsverlust an der letzten Schicht hält Scheibe und Quadrat an ihrem Platz.

# %%
result = style_transfer(style_image, content_image, style_layers, content_layers=[-1],
                        style_weights=style_weights, content_weights=[1e-1])
show_images([content_image, style_image, result], ["Inhaltsbild", "Stilbild", "Ergebnis"])

# %% [markdown]
# Scheibe und Quadrat bleiben erkennbar, Fläche und Hintergrund bekommen buntes Korn und schräge
# Farbsäume aus dem Stilbild. Mit Zufallsgewichten wird das Bild körniger als mit einem
# trainierten VGG19.

# %% [markdown]
# ## 5. Layer und Gewichte variieren
#
# In Abschnitt 4 waren Schichten und Gewichte fest vorgegeben. Genau diese Einstellungen
# sollen Sie in Aufgabe 1 verändern. Die folgende Zelle rechnet fünf Varianten, jeweils mit
# dem Inhaltsbild als Startbild; die Einträge in `variants` können Sie ändern oder ergänzen.

# %%
# Hier ändern: Name -> (style_layers, style_weights, content_weights)
variants = {
    "Inhalt 1": ([1, 3, 6, 8, 11], style_weights, [1e0]),
    "Inhalt 0,1": ([1, 3, 6, 8, 11], style_weights, [1e-1]),
    "Inhalt 0,01": ([1, 3, 6, 8, 11], style_weights, [1e-2]),
    "nur frühe Stil-Schichten": ([1, 3], style_weights[:2], [1e-1]),
    "nur späte Stil-Schichten": ([8, 11], style_weights[3:], [1e-1]),
}

results = {}
for name, (layers, s_weights, c_weights) in variants.items():
    results[name] = style_transfer(style_image, content_image, layers, [-1], s_weights, c_weights)
show_images(list(results.values()), list(results.keys()))

# %% [markdown]
# Mit Inhalts-Gewicht 1 bleiben die Formen klar und vom Stil kommt nur ein Farbkorn an, mit
# 0,1 legen sich Streifen um die Formen, mit 0,01 beherrschen die Streifen das Bild. Es kommt
# also auf das Verhältnis von Inhalts- zu Stil-Gewichten an. Mit den frühen Stil-Schichten
# allein kommen Farben und kurze Kanten an, mit den späten beim Ersatznetz fast nur Korn. Bei
# einem trainierten VGG19 erfassen späte Schichten größere Strukturen; probieren Sie das in
# der Aufgabe aus.
#
# Welche Gewichte passen, hängt auch von der Größenordnung der Bildwerte ab: Doppelte
# Bildwerte machen den Inhaltsverlust etwa viermal, den Stilverlust etwa sechzehnmal so groß,
# weil die Gram-Matrix zwei Ausgaben multipliziert und der MSE noch einmal quadriert. Die
# Gewichte der Aufgabe sind auf `preprocessing` abgestimmt, das die Bildwerte mit 255
# multipliziert. Unsere Ersatzbilder liegen zwischen 0 und 1 und brauchen deshalb andere
# Gewichte.
#
# Ohne GPU dauert in der Aufgabe jede Auswertung einige Sekunden. Zum Ausprobieren können Sie
# `maximum_iterations` auf 200 bis 300 senken; der Stil ist dann schon deutlich zu sehen.

# %% [markdown]
# # Teil B: DCGAN

# %%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm  # Textbalken, braucht kein Widget-JavaScript

print(f"PyTorch {torch.__version__}, Rechnen auf: {'GPU' if torch.cuda.is_available() else 'CPU'}")

# %% [markdown]
# ## 6. Formen durch Generator und Diskriminator
#
# Beim Style Transfer stand das Netz fest. Beim GAN werden dagegen zwei Netze trainiert, ein
# Generator und ein Diskriminator. Das Buch vergleicht sie mit einem Fälscher, der aus einem
# Zufallsvektor ein Bild macht, und der Polizei, die schätzt, ob ein Bild echt ist. Der
# Generator beginnt mit einem Vektor der Form `(nz, 1, 1)`, einem "Bild" mit `nz` Kanälen und
# einem Pixel. `ConvTranspose2d`, die transponierte Faltung, vergrößert es schrittweise. Wie sie
# Werte verteilt, zeigt ein 2 × 2-Eingang mit einem 3 × 3-Kernel aus Einsen und Stride 2:

# %%
deconv = nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, bias=False)
nn.init.ones_(deconv.weight)  # alle Kernelgewichte 1, damit man die Summen sieht

with torch.no_grad():
    # view(1, 1, 2, 2): Batch- und Kanalachse ergänzen, wie es Faltungsschichten erwarten
    one_pixel = torch.tensor([[1.0, 0.0], [0.0, 0.0]]).view(1, 1, 2, 2)  # nur oben links eine 1
    all_pixels = torch.ones(1, 1, 2, 2)
    print("Eingang: nur oben links eine 1")
    print(deconv(one_pixel)[0, 0])  # [0, 0]: erstes Bild, erster Kanal -> 5 × 5 Werte
    print("Eingang: alle vier Pixel 1")
    print(deconv(all_pixels)[0, 0])

# %% [markdown]
# Jedes Eingangspixel stempelt den Kernel, mit seinem Wert multipliziert, auf die Ausgabe. Der
# Stride gibt an, wie weit die Stempel auseinanderliegen: Bei Stride 2 beginnt der Stempel des
# nächsten Pixels zwei Positionen weiter. Wo sich zwei Stempel überlappen, werden die Werte
# addiert, deshalb steht in der Mitte eine 4, und aus 2 × 2 wird 5 × 5. Die Kernelwerte werden
# wie bei jeder Faltung gelernt; eine Umkehrung der Faltung ist das nicht (siehe z04,
# Abschnitt 4). Generator und Diskriminator der Aufgabe verwenden Kernelgröße 4 in diesen
# Einstellungen:

# %%
first = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=1, padding=0)  # erste Generatorschicht
later = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)  # alle weiteren
down = nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1)  # Diskriminator, erste vier
last = nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=0)  # Diskriminator, letzte

print("ConvTranspose2d(4, stride=1, padding=0):  1 ->", first(torch.rand(1, 1, 1, 1)).shape[-1])
for side in [4, 8, 16, 32]:
    print(f"ConvTranspose2d(4, stride=2, padding=1): {side:>2} -> {later(torch.rand(1, 1, side, side)).shape[-1]}")
for side in [64, 32, 16, 8]:
    print(f"Conv2d(4, stride=2, padding=1):          {side:>2} -> {down(torch.rand(1, 1, side, side)).shape[-1]}")
print("Conv2d(4, stride=1, padding=0):           4 ->", last(torch.rand(1, 1, 4, 4)).shape[-1])


# %% [markdown]
# Die erste Schicht des Generators macht aus einem Pixel 4 × 4, jede weitere mit Stride 2 und
# Padding 1 verdoppelt die Größe. `Conv2d` mit denselben Einstellungen halbiert sie, die letzte
# macht aus 4 × 4 ein Pixel. Damit können Sie für Aufgabe 2 nachzählen, wie viele Schichten bis
# 64 × 64 nötig sind. Unsere Ersatznetze für 32 × 32 Pixel sind eine Schicht kürzer, und der
# Diskriminator hat zunächst wie im Buch keine BatchNorm nach der ersten Faltung:

# %%
class SmallGenerator(nn.Module):
    def __init__(self, nz, ngf, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),  # 1 -> 4
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),  # 4 -> 8
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),  # 8 -> 16
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),  # 16 -> 32, nc Farbkanäle
            nn.Tanh(),  # Werte von -1 bis 1 wie die normierten echten Bilder (Abschnitt 8)
        )

    def forward(self, input):
        return self.main(input)


class SmallDiscriminator(nn.Module):
    def __init__(self, ndf, nc=3, first_batchnorm=False):
        super().__init__()
        # BatchNorm nach der ersten Faltung: wie in der Aufgabe, ohne: wie im Buch (Abschnitt 11)
        first_norm = [nn.BatchNorm2d(ndf)] if first_batchnorm else []
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # 32 -> 16
            *first_norm,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 16 -> 8
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 8 -> 4
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),  # 4 -> 1, ein Wert je Bild
            nn.Sigmoid(),  # Wert zwischen 0 und 1: Wahrscheinlichkeit für "echt"
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)  # (Batch, 1, 1, 1) -> (Batch,)


nz = 32  # Länge des Zufallsvektors, in der Aufgabe 100
generator = SmallGenerator(nz, ngf=32)
discriminator = SmallDiscriminator(ndf=32)

h = torch.randn(16, nz, 1, 1)  # 16 Zufallsvektoren
print(f"Zufallsvektor:     {tuple(h.shape)} = 16 Vektoren, {nz} Kanäle, 1 × 1")
for layer in generator.main:
    h = layer(h)
    if isinstance(layer, nn.ConvTranspose2d):
        print(f"ConvTranspose2d -> {tuple(h.shape)}")
for layer in discriminator.main:
    h = layer(h)
    if isinstance(layer, nn.Conv2d):
        print(f"Conv2d          -> {tuple(h.shape)}")
print(f"nach forward:      {tuple(discriminator(generator(torch.randn(16, nz, 1, 1))).shape)} "
      f"= je Bild eine Zahl")

# %% [markdown]
# Der Generator geht von 1 über 4, 8 und 16 auf 32 Pixel, die Kanalzahl fällt dabei von `4 *
# ngf` auf 3 Farbkanäle; der Diskriminator geht denselben Weg zurück. Am Ende steht `(16, 1,
# 1, 1)`, also 16 Bilder mit je einem Kanal und einem Pixel. `view(-1, 1).squeeze(1)` macht
# daraus 16 Zahlen, denn die Verlustfunktion erwartet Ausgabe und Labels in derselben Form,
# und die Labels der Aufgabe haben die Form `(b_size,)`. Passen Kanalzahlen oder Formen nicht
# zueinander, bricht der erste Durchlauf mit einer Fehlermeldung ab. Drei typische Meldungen:

# %%
criterion = nn.BCELoss()  # die Verlustfunktion aus dem Buch, Abschnitt 8 erklärt sie
attempts = {
    # Kanalzahl der Schicht passt nicht zur Ausgabe der vorigen Schicht (128 statt 64)
    "falsche Kanalzahl": lambda: nn.ConvTranspose2d(64, 32, 4, 2, 1)(torch.rand(16, 128, 4, 4)),
    # Ausgabe (16, 1, 1, 1) ohne view/squeeze mit Labels der Form (16,) verglichen
    "falsche Form der Ausgabe": lambda: criterion(torch.rand(16, 1, 1, 1), torch.ones(16)),
    # Diskriminator ohne Sigmoid am Ende: Ausgaben außerhalb von 0 bis 1
    "Werte außerhalb von 0 bis 1": lambda: criterion(torch.randn(16), torch.ones(16)),
}
for name, attempt in attempts.items():
    try:
        attempt()
    except (RuntimeError, ValueError) as error:
        print(f"{name}:\n   {type(error).__name__}: {error}\n")

# %% [markdown]
# Die erste nennt erwartete und tatsächliche Kanalzahl; dann passt `in_channels` einer Schicht
# nicht zu `out_channels` der vorigen. Die zweite bedeutet, dass die Ausgabe nicht auf die Form
# der Labels gebracht wurde, die dritte, dass die Sigmoid-Funktion am Ende fehlt.

# %% [markdown]
# ## 7. BatchNorm, Initialisierung und LeakyReLU
#
# Neben den Faltungen aus Abschnitt 6 stecken in beiden Netzen BatchNorm, eine eigene
# Initialisierung und im Diskriminator LeakyReLU. `BatchNorm2d` normiert jeden Kanal innerhalb
# des Batches auf Mittelwert 0 und Streuung 1 und wendet danach eine lernbare Skalierung und
# Verschiebung an, anfangs 1 und 0. Das Buch begründet BatchNorm damit, dass zwar die
# Eingabedaten normiert sind, die Werte in den Zwischenschichten aber nicht mehr:

# %%
torch.manual_seed(0)
values = 5 + 3 * torch.randn(64, 1, 4, 4)  # Batch mit Mittelwert etwa 5 und Streuung etwa 3
batchnorm = nn.BatchNorm2d(1)  # ein Kanal; neu erzeugt ist die Schicht im Trainingsmodus

normalized = batchnorm(values)  # im Trainingsmodus
print(f"vorher:                 Mittelwert {values.mean():.2f}, Streuung {values.std():.2f}")
print(f"nach BatchNorm (train): Mittelwert {normalized.mean():.2f}, Streuung {normalized.std():.2f}")


# %% [markdown]
# Mittelwert und Streuung des Batches liegen danach genau bei 0 und 1. Im Auswertungsmodus
# (`eval()`) verwendet BatchNorm statt der Werte des aktuellen Batches Mittelwerte, die es
# während des Trainings mitführt (wie in z04, Abschnitt 5). Aufgabe 2 erzeugt die Bilder aus
# `fixed_noise` ohne `eval()`; das ist bei DCGANs üblich und funktioniert, weil `fixed_noise`
# selbst ein ganzer Batch ist.
#
# `weights_inititialisation` aus der Aufgabe zieht die Faltungsgewichte aus einer
# Normalverteilung mit Streuung 0,02 wie im DCGAN-Paper und die Skalierung von BatchNorm aus
# einer Normalverteilung um 1; `net.apply(...)` ruft die Funktion für jede Schicht auf.

# %%
def weights_inititialisation(m):  # wie in der Aufgabe
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:  # trifft Conv2d und ConvTranspose2d
        m.weight.data.normal_(0.0, 0.02)  # Mittelwert 0, Streuung 0,02
    elif class_name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)  # Skalierung um 1
        m.bias.data.fill_(0)  # Verschiebung 0


torch.manual_seed(0)
generator = SmallGenerator(nz, ngf=32)
print(f"vorher:  Streuung der ersten ConvTranspose2d-Gewichte {generator.main[0].weight.std():.4f}")
generator.apply(weights_inititialisation)
print(f"nachher: Streuung der ersten ConvTranspose2d-Gewichte {generator.main[0].weight.std():.4f}")
print(f"BatchNorm-Skalierung: Mittelwert {generator.main[1].weight.mean():.3f}")

# %% [markdown]
# Vorher hängt die Streuung der Gewichte von der Größe der Schicht ab, danach haben alle
# Faltungen die Streuung 0,02. Im Diskriminator steht statt `ReLU` eine `LeakyReLU(0.2)`:

# %%
values = torch.tensor([-3.0, -1.0, 0.0, 2.0])
print("Eingabe:        ", values.tolist())
print("ReLU:           ", torch.relu(values).tolist())
print("LeakyReLU(0.2): ", [round(v, 2) for v in nn.LeakyReLU(0.2)(values).tolist()])

# %% [markdown]
# Negative Werte werden mit 0,2 multipliziert statt auf 0 gesetzt. So bleibt ein kleiner
# Gradient, auch wenn eine Einheit nur negative Eingaben bekommt. Das ist hier besonders
# wichtig, weil der Generator nur über die Gradienten lernt, die durch den Diskriminator
# zurückfließen.

# %% [markdown]
# ## 8. Die Verlustfunktion
#
# Für das Training fehlt noch die Verlustfunktion, nach der Aufgabe 2 mit einem "Warum?" fragt.
# Der Diskriminator endet mit einer Sigmoid-Funktion und gibt eine Wahrscheinlichkeit für "echt"
# aus, die Labels sind 1 und 0. Er ist also ein Klassifikator mit zwei Klassen wie eine
# logistische Regression. Das Buch verwendet dafür die binäre Kreuzentropie `nn.BCELoss()`:

# %%
criterion = nn.BCELoss()
print("Ausgabe D   Label 1 (echt)   Label 0 (fake)")
for p in [0.99, 0.9, 0.5, 0.1, 0.01]:
    output = torch.tensor([p])
    print(f"   {p:4.2f}        {criterion(output, torch.tensor([1.0])).item():6.3f}"
          f"           {criterion(output, torch.tensor([0.0])).item():6.3f}")

# %% [markdown]
# Liegt die Ausgabe nahe am Label, ist der Verlust klein; bei einem sicheren Fehlurteil wie
# 0,01 statt 1 steigt er auf etwa 4,6, während `MSELoss` dort nur knapp 1 liefern würde.
# Sichere Fehlurteile werden also besonders hart bestraft. Bei 0,5 ist er für beide Labels
# etwa 0,69. Ein ratender Diskriminator hat damit `errD` von etwa 1,39 (echte plus gefälschte
# Bilder) und der Generator `errG` von etwa 0,69; das hilft beim Lesen der Kurven in Abschnitt
# 10.
#
# Auch beim Generator ist die letzte Funktion auf die Daten abgestimmt: Die echten Bilder
# bringt `Normalize` auf −1 bis 1, deshalb endet der Generator mit `Tanh`, das genau diesen
# Bereich liefert.

# %% [markdown]
# ## 9. Die Trainingsschleife zerlegt
#
# Mit der Verlustfunktion aus Abschnitt 8 sind alle Teile beisammen. Die Schleife der Aufgabe
# ist lang, weil in jeder Iteration zwei Netze nacheinander lernen. Wir gehen deshalb eine
# Iteration Schritt für Schritt durch. Die Aufgabe trainiert auf CIFAR10, einem Datensatz mit
# 60.000 kleinen Farbfotos aus zehn Klassen wie Flugzeugen, Autos und Tieren; ohne GPU dauert
# das Stunden. Hier nehmen wir stattdessen Ersatzbilder: 32 × 32 Pixel, eine rote, grüne oder
# blaue Scheibe auf grauem Grund, jede Farbe in einem Drittel der Bilder. Die Farbe bekommt das
# GAN nie zu sehen; sie dient in Abschnitt 10 nur zur Kontrolle.

# %%
from torch.utils.data import DataLoader, TensorDataset

COLOR_NAMES = ["rot", "grün", "blau"]
DISC_COLORS = torch.tensor([[1.0, 0.15, 0.15], [0.15, 0.9, 0.15], [0.2, 0.35, 1.0]])


def make_disc_images(n_images, seed):
    generator_rng = torch.Generator().manual_seed(seed)  # eigene Zufallsquelle, feste Bilder
    ys, xs = torch.meshgrid(torch.arange(32.0), torch.arange(32.0), indexing="ij")
    images = torch.zeros(n_images, 3, 32, 32)
    colors = torch.randint(3, (n_images,), generator=generator_rng)  # 0 rot, 1 grün, 2 blau
    for k in range(n_images):
        center_x, center_y = (16 + torch.empty(2).uniform_(-8, 8, generator=generator_rng)).tolist()
        radius = torch.empty(1).uniform_(4, 8, generator=generator_rng).item()
        background = torch.empty(1).uniform_(0.2, 0.6, generator=generator_rng).item()  # Grauwert
        # mask: 1 innerhalb der Scheibe, 0 außerhalb, am Rand ein weicher Übergang
        mask = torch.sigmoid(-((xs - center_x) ** 2 + (ys - center_y) ** 2).sqrt() + radius)
        images[k] = background * (1 - mask) + DISC_COLORS[colors[k]][:, None, None] * mask
    return images * 2 - 1, colors  # wie Normalize((0.5, ...), (0.5, ...)): 0..1 -> -1..1


real_images, real_colors = make_disc_images(4000, seed=1)
print(f"real_images: {tuple(real_images.shape)} = 4000 Bilder, 3 Farbkanäle, 32 × 32 Pixel")
print(f"Wertebereich: {real_images.min():.2f} bis {real_images.max():.2f}")
print("Anteil der Farben:", {name: f"{(real_colors == c).float().mean():.0%}"
                             for c, name in enumerate(COLOR_NAMES)})

batch_size = 64
dataloader = DataLoader(TensorDataset(real_images, real_colors), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(0))

import torchvision.utils as vutils


def show_grid(images, title, nrow=8):
    grid = vutils.make_grid(images[:nrow * nrow], nrow=nrow, padding=2, normalize=True)  # -> 0..1
    plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title(title)
    plt.axis("off")
    plt.show()


show_grid(real_images, "echte Bilder (Ersatz für CIFAR10)")

# %% [markdown]
# Jedes Bild zeigt eine Scheibe in einer der drei Farben; Lage, Größe und die Helligkeit des
# Hintergrunds ändern sich. Netze, Initialisierung, Verlust und Optimierer sind wie in der
# Aufgabe aufgesetzt, nur mit den kleineren Netzen:

# %%
torch.manual_seed(0)
generator = SmallGenerator(nz, ngf=32)
generator.apply(weights_inititialisation)
discriminator = SmallDiscriminator(ndf=32)
discriminator.apply(weights_inititialisation)

criterion = nn.BCELoss()
lr, beta1 = 0.0002, 0.5  # wie in der Aufgabe
# Zwei getrennte Optimierer: jeder verändert nur die Gewichte seines eigenen Netzes
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr, betas=(beta1, 0.95))
optimizer_generator = torch.optim.Adam(generator.parameters(), lr, betas=(beta1, 0.95))
real_label, fake_label = 1, 0

data, _ = next(iter(dataloader))  # ein Batch echter Bilder; die Farbe wird ignoriert
b_size = data.size(0)

# (1a) Diskriminator auf echten Bildern: Ziel ist Label 1
discriminator.zero_grad()  # alte Gradienten des Diskriminators löschen
label = torch.full((b_size,), real_label, dtype=torch.float)  # Vektor aus b_size Einsen
output = discriminator(data)  # je Bild eine Wahrscheinlichkeit für "echt"
errD_real = criterion(output, label)
errD_real.backward()
D_x = output.mean().item()  # mittleres Urteil über die echten Bilder als Python-Zahl

# (1b) Diskriminator auf gefälschten Bildern: Ziel ist Label 0
noise = torch.randn(b_size, nz, 1, 1)  # je Bild ein Zufallsvektor aus der Normalverteilung
fake = generator(noise)
label.fill_(fake_label)  # alle Einträge auf 0 setzen; _ am Ende: im selben Tensor
output = discriminator(fake.detach())  # detach: kein Gradient zurück in den Generator
errD_fake = criterion(output, label)
errD_fake.backward()  # addiert sich zu den Gradienten aus (1a)
D_G_z1 = output.mean().item()
print(f"nach (1b): Gradient im Generator: {generator.main[0].weight.grad}")
optimizer_discriminator.step()  # nur der Diskriminator ändert sich

# (2) Generator: seine Fälschungen sollen als echt (Label 1) durchgehen
generator.zero_grad()  # alte Gradienten des Generators löschen
label.fill_(real_label)
output = discriminator(fake)  # diesmal ohne detach, der Gradient soll zum Generator
errG = criterion(output, label)
errG.backward()
D_G_z2 = output.mean().item()
print(f"nach (2):  Gradient im Generator vorhanden: {generator.main[0].weight.grad is not None}")
print(f"nach (2):  Gradient im Diskriminator vorhanden: {discriminator.main[0].weight.grad.abs().sum() > 0}")
optimizer_generator.step()  # nur der Generator ändert sich

print(f"D(x) = {D_x:.3f}   D(G(z)) vor dem Diskriminator-Schritt = {D_G_z1:.3f}   "
      f"danach = {D_G_z2:.3f}")
print(f"errD = {errD_real.item() + errD_fake.item():.3f}   errG = {errG.item():.3f}")

# %% [markdown]
# Die Ausgabe zeigt, wer wann einen Gradienten bekommt:
#
# 1. Nach (1b) hat der Generator keinen Gradienten: `fake.detach()` schneidet den Weg zurück ab,
#    denn laut Buch soll der Generator nicht lernen, während der Diskriminator trainiert wird.
# 2. In (2) fließt der Gradient bis in den Generator und auch in den Diskriminator; verändert
#    wird aber nur der Generator. Die liegengebliebenen Gradienten löscht in der Aufgabe
#    `net_discriminator.zero_grad()` zu Beginn der nächsten Iteration.
# 3. Das Label in (2) ist 1: Der Generator wird bestraft, wenn der Diskriminator seine Bilder
#    als Fälschung erkennt.
#
# `D_x`, `D_G_z1` und `D_G_z2` sind die Werte aus dem Fortschrittsbalken der Aufgabe: das
# mittlere Urteil über echte Bilder, über die Fälschungen vor dem Schritt des Diskriminators
# und danach. Das Urteil über dieselben Fälschungen sinkt nach einem Schritt des
# Diskriminators von etwa 0,40 auf 0,30; `errD` liegt mit 1,37 noch nahe am Wert eines
# ratenden Diskriminators. Was ohne `detach()` passiert:

# %%
generator.zero_grad()  # Gradienten aus der Iteration oben löschen, damit man nur die neuen sieht
fake = generator(torch.randn(b_size, nz, 1, 1))
label.fill_(fake_label)
errD_fake = criterion(discriminator(fake), label)  # ohne detach
errD_fake.backward()
print(f"Gradient im Generator nach (1b) ohne detach vorhanden: {generator.main[0].weight.grad.abs().sum() > 0}")

generator.zero_grad()  # wie in der Aufgabe vor dem Schritt des Generators
label.fill_(real_label)
try:
    criterion(discriminator(fake), label).backward()
except RuntimeError as error:
    print(f"Schritt (2): RuntimeError: {str(error).split(' (')[0]}")
discriminator.zero_grad()  # aufräumen, die Netze werden hier nicht weiter benutzt
generator.zero_grad()


# %% [markdown]
# Dass der Generator dabei einen Gradienten bekommt, schadet allein noch nicht, denn
# `zero_grad()` löscht ihn vor seinem Schritt. Das eigentliche Problem zeigt die zweite Zeile:
# PyTorch gibt die Zwischenergebnisse nach dem ersten `backward()` frei, deshalb scheitert das
# zweite `backward()` durch denselben Durchlauf des Generators. Mit `detach()` endet das erste
# `backward()` am Diskriminator, und der Weg durch den Generator bleibt für Schritt (2)
# erhalten.

# %% [markdown]
# ## 10. Training und Deutung der Kurven
#
# Jetzt läuft die Schleife aus Abschnitt 9 über alle Batches und zehn Epochen; ohne GPU dauert
# das knapp eine Minute. Statt der Animation der Aufgabe, die JavaScript braucht, zeigen wir die
# Bilder aus `fixed_noise` zu einigen Zeitpunkten. Am Ende zählen wir, welche Farbe 600 erzeugte
# Bilder haben.

# %%
def dominant_color(images):
    images = (images + 1) / 2  # zurück auf 0 bis 1
    # Mittelwert über die drei Farbkanäle ist der Grauanteil; was darüber liegt, ist Farbe
    chroma = images - images.mean(dim=1, keepdim=True)
    # nur positive Abweichungen über alle Pixel summieren, dann der stärkste Kanal je Bild
    return chroma.clamp(min=0).sum(dim=(2, 3)).argmax(dim=1)  # 0 rot, 1 grün, 2 blau


def train_gan(ngf=32, ndf=32, num_epochs=10, first_batchnorm=False, seed=0):
    torch.manual_seed(seed)
    net_generator = SmallGenerator(nz, ngf)
    net_generator.apply(weights_inititialisation)
    net_discriminator = SmallDiscriminator(ndf, first_batchnorm=first_batchnorm)
    net_discriminator.apply(weights_inititialisation)
    criterion = nn.BCELoss()
    optimizer_discriminator = torch.optim.Adam(net_discriminator.parameters(), lr, betas=(beta1, 0.95))
    optimizer_generator = torch.optim.Adam(net_generator.parameters(), lr, betas=(beta1, 0.95))
    fixed_noise = torch.randn(64, nz, 1, 1)  # immer dieselben 64 Zufallsvektoren
    loader = DataLoader(TensorDataset(real_images, real_colors), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))

    history = {"G": [], "D": [], "D(x)": [], "D(G(z))": []}
    snapshots = []  # (Iteration, Bilder aus fixed_noise)
    iters = 0
    for epoch in tqdm(range(num_epochs)):
        for data, _ in loader:
            # (1) Diskriminator: echte Bilder -> 1, Fälschungen -> 0
            net_discriminator.zero_grad()
            b_size = data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float)
            output = net_discriminator(data)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            fake = net_generator(torch.randn(b_size, nz, 1, 1))
            label.fill_(fake_label)
            output = net_discriminator(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            optimizer_discriminator.step()

            # (2) Generator: Fälschungen sollen als echt gelten
            net_generator.zero_grad()
            label.fill_(real_label)
            output = net_discriminator(fake)
            errG = criterion(output, label)
            errG.backward()
            optimizer_generator.step()

            history["G"].append(errG.item())
            history["D"].append(errD_real.item() + errD_fake.item())
            history["D(x)"].append(D_x)
            history["D(G(z))"].append(D_G_z1)
            if iters % 125 == 0:
                with torch.no_grad():
                    snapshots.append((iters, net_generator(fixed_noise)))
            iters += 1
    with torch.no_grad():
        snapshots.append((iters, net_generator(fixed_noise)))
        samples = net_generator(torch.randn(600, nz, 1, 1))
    # bincount zählt, wie oft 0, 1 und 2 vorkommen, also wie viele Bilder je Farbe
    counts = torch.bincount(dominant_color(samples), minlength=3).tolist()
    return history, snapshots, counts


history, snapshots, counts = train_gan()
print(f"{len(history['G'])} Iterationen")
print("erzeugte Bilder je Farbe:", dict(zip(COLOR_NAMES, counts)))

# %% [markdown]
# Die drei Farben kommen etwa gleich oft vor, wie bei den echten Bildern. Links stehen die
# Verluste mit einem gleitenden Mittelwert, grau gepunktet und gestrichelt die Werte eines
# ratenden Diskriminators aus Abschnitt 8, rechts `D(x)` und `D(G(z))`:

# %%
import numpy as np


def smooth(values, window=20):  # gleitender Mittelwert, damit der Verlauf lesbar wird
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_history(history, title):
    fig, (ax_loss, ax_prob) = plt.subplots(1, 2, figsize=(11, 3.5))
    ax_loss.plot(history["G"], color="tab:blue", alpha=0.25)
    ax_loss.plot(smooth(history["G"]), color="tab:blue", label="G (errG)")
    ax_loss.plot(history["D"], color="tab:orange", alpha=0.25)
    ax_loss.plot(smooth(history["D"]), color="tab:orange", label="D (errD)")
    ax_loss.axhline(0.69, color="gray", linestyle=":")  # errG eines ratenden Diskriminators
    ax_loss.axhline(1.39, color="gray", linestyle="--")  # errD eines ratenden Diskriminators
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Verlust")
    ax_loss.legend()
    ax_prob.plot(smooth(history["D(x)"]), label="D(x): echte Bilder")
    ax_prob.plot(smooth(history["D(G(z))"]), label="D(G(z)): Fälschungen")
    ax_prob.axhline(0.5, color="gray", linestyle=":")
    ax_prob.set_ylim(0, 1)
    ax_prob.set_xlabel("Iteration")
    ax_prob.set_ylabel("mittlere Ausgabe des Diskriminators")
    ax_prob.legend()
    fig.suptitle(title)
    plt.show()


def show_snapshots(snapshots, count=4):
    picks = [snapshots[k] for k in np.linspace(0, len(snapshots) - 1, count).astype(int)]
    fig, axes = plt.subplots(1, count, figsize=(3.2 * count, 3.4))
    for ax, (iteration, images) in zip(axes, picks):
        grid = vutils.make_grid(images[:16], nrow=4, padding=2, normalize=True)
        ax.imshow(grid.permute(1, 2, 0))
        ax.set_title(f"nach {iteration} Iterationen")
        ax.axis("off")
    plt.show()


plot_history(history, "Ausgewogene Netze: ngf = ndf = 32")
show_snapshots(snapshots)

# %% [markdown]
# Anders als in allen bisherigen Terminen fällt keine Kurve einfach ab. Zu Beginn gewinnt der
# Diskriminator: `D(x)` steigt fast auf 1, `D(G(z))` fällt fast auf 0, und `errG` steigt auf
# über 5. Danach schwanken die Kurven, weil abwechselnd das eine und das andere Netz aufholt.
# Am Ende liegt `errG` bei etwa 3; das Gleichgewicht, in dem der Diskriminator nur noch rät,
# ist in dieser kurzen Zeit nicht erreicht. Die Bilder zeigen den Fortschritt besser: nach 500
# Iterationen erste Scheiben, am Ende viele deutliche, aber auch einige mit zwei Scheiben oder
# nur einem blassen Fleck.
#
# Eine Testgenauigkeit wie bei einer Klassifikation gibt es beim GAN nicht. Achten Sie in der
# Aufgabe deshalb vor allem auf die Bilder und auf ein Warnzeichen: Bleiben `errD` oder
# `D(G(z))` dauerhaft nahe 0, hat der Diskriminator gewonnen.

# %% [markdown]
# ## 11. Wenn ein Netz zu stark wird
#
# Aufgabe 2 weist darauf hin, dass ein deutlich besserer Diskriminator ungünstig ist: Erkennt er
# jede Fälschung sicher, bekommt der Generator kaum brauchbare Rückmeldung. Wir testen zwei
# Änderungen, die das Gleichgewicht verschieben können: einen kleineren Generator und eine
# zusätzliche BatchNorm nach der ersten Faltung des Diskriminators, wie sie die Aufgabe
# beschreibt. Die Zelle braucht ohne GPU etwa zwei Minuten.

# %%
# Hier ändern: Name -> Einstellungen für train_gan (fehlende Werte wie in Abschnitt 10)
gan_variants = {
    "ngf = 8: schwacher Generator": dict(ngf=8),
    "BatchNorm auch nach der ersten Diskriminator-Schicht": dict(first_batchnorm=True),
}

gan_results = {}
for name, settings in gan_variants.items():
    gan_results[name] = train_gan(**settings)
    variant_history, _, variant_counts = gan_results[name]
    last = {key: np.mean(values[-100:]) for key, values in variant_history.items()}
    print(f"{name}\n   letzte 100 Iterationen: errD {last['D']:.2f}, errG {last['G']:.2f}, "
          f"D(x) {last['D(x)']:.2f}, D(G(z)) {last['D(G(z))']:.2f}; "
          f"Farben: {dict(zip(COLOR_NAMES, variant_counts))}")

last = {key: np.mean(values[-100:]) for key, values in history.items()}
print(f"zum Vergleich ausgewogen (Abschnitt 10)\n   letzte 100 Iterationen: errD {last['D']:.2f}, "
      f"errG {last['G']:.2f}, D(x) {last['D(x)']:.2f}, D(G(z)) {last['D(G(z))']:.2f}; "
      f"Farben: {dict(zip(COLOR_NAMES, counts))}")

# %%
fig, axes = plt.subplots(1, len(gan_results), figsize=(3.6 * len(gan_results), 3.9))
for ax, (name, (_, variant_snapshots, _)) in zip(axes, gan_results.items()):
    grid = vutils.make_grid(variant_snapshots[-1][1][:16], nrow=4, padding=2, normalize=True)
    ax.imshow(grid.permute(1, 2, 0))
    ax.set_title(name.replace(": ", ":\n"), fontsize=9)
    ax.axis("off")
plt.show()

# %% [markdown]
# - Beim kleineren Generator (`ngf = 8`) gewinnt der Diskriminator: `errD` liegt bei 0,08,
#   `D(x)` bei 0,99, `D(G(z))` bei 0,01. Die Bilder sind gemusterte Flächen in Grün und Violett,
#   Blau fehlt fast ganz.
# - Mit BatchNorm nach der ersten Faltung gewinnt bei unseren kleinen Netzen der Diskriminator
#   ebenso deutlich, `errD` liegt bei 0,02. Der Generator erzeugt grünliche und graue Flächen und kaum eine blaue.
#
# In beiden Fällen deckt der Generator nur noch einen Teil der Vielfalt der echten Bilder ab.
# Im Extremfall entsteht Mode Collapse: Der Generator erzeugt nur noch wenige, fast gleiche
# Bilder.
#
# Das Ergebnis eines GAN-Trainings hängt stark von der zufälligen Initialisierung ab; mit
# einem anderen `seed` kann eine Variante anders ausgehen. Liefert Ihr DCGAN in der Aufgabe
# keine brauchbaren Bilder und bleibt `D(G(z))` nahe 0, verschieben Sie das Gleichgewicht, etwa
# über das Verhältnis von `ngf` zu `ndf`.

# %% [markdown]
# ## 12. Zusammenfassung
#
# - Layer-Nummern sind Positionen in `vgg19().features`; das `n` in `1e3/n**2` ist die Kanalzahl
#   der Schicht.
# - Hooks schneiden die Ausgaben einzelner Schichten mit.
# - Die Gram-Matrix misst, welche Kanäle gemeinsam aktiv sind, unabhängig vom Ort; darum ist sie
#   das Maß für den Stil.
# - Beim Style Transfer wird das Bild optimiert; L-BFGS wertet den Verlust pro Schritt mehrmals
#   über `closure` aus.
# - Das Verhältnis von Inhalts- zu Stil-Gewichten bestimmt, wie viel von den Formen bleibt.
# - Transponierte Faltungen verdoppeln mit Stride 2 die Bildgröße, Faltungen halbieren sie.
# - BatchNorm normiert je Kanal im Batch; LeakyReLU lässt negative Werte mit Faktor 0,2 durch.
# - Der Diskriminator ist ein Klassifikator mit zwei Klassen, daher die binäre Kreuzentropie.
# - `detach()` hält den Generator aus dem Schritt des Diskriminators heraus.
# - GAN-Verluste schwanken; aussagekräftiger sind die Bilder und das Warnzeichen `D(G(z))`
#   nahe 0.
# - Ein zu starker Diskriminator lässt den Generator Vielfalt verlieren.
