# %% [markdown]
# # Ergänzung zu Termin 4: Autoencoder und Variational Autoencoder
#
# In der Aufgabe zu Termin 4 trainieren Sie zwei Netze auf FashionMNIST, die ein Bild nicht
# klassifizieren, sondern wiederherstellen sollen. Im ersten Teil ist es ein Autoencoder: Ein
# Encoder drückt jedes 28×28-Bild auf wenige Zahlen zusammen, die Größe `latent_size`, und ein
# Decoder baut daraus wieder ein Bild. Im zweiten Teil wird daraus ein Variational Autoencoder
# (VAE), dessen Encoder für jedes Bild eine Normalverteilung liefert, aus der ein zufälliger
# Punkt erzeugt wird. Für beide Teile müssen Sie Entscheidungen treffen, die das
# Aufgaben-Notebook nur benennt: wie klein `latent_size` sein darf, welche Aktivierung am Ende
# des Decoders steht, welche Verlustfunktion passt, wie `reparametrize` funktioniert und welchen
# Wert `beta` bekommt. Kapitel 6 des Buchs liefert dazu die Bausteine: den Engpass, den
# Convolutional Autoencoder, den VAE mit dem Reparametrisierungs-Trick und den KL-Term in der
# Verlustfunktion.
#
# Wir folgen dem Weg eines Bildes durch die beiden Netze; rechts steht jeweils, in welchem
# Abschnitt und zu welchem Teil der Aufgabe der Schritt behandelt wird:
#
# ```
# Bild (1, 28, 28), Werte 0 bis 1                       Abschnitt 1
#         │  Encoder: Conv2d / Linear                   Abschnitt 4, Teil 1
#         ▼
# latenter Vektor, latent_size Zahlen                   Engpass: Abschnitt 2, Teil 1
#         │                                             latenter Raum: Abschnitte 5 und 6
#         │  beim VAE stattdessen: mu und logvar,
#         │  daraus mit reparametrize ein Zufallspunkt  Abschnitt 7, Teil 2
#         ▼
#         │  Decoder: Linear / ConvTranspose2d          Abschnitt 4, Teil 1
#         ▼
# Rekonstruktion (1, 28, 28)                            letzte Aktivierung: Abschnitt 3, Teil 1
#         │  Vergleich mit dem Eingabebild
#         ▼
# Verlust: Rekonstruktion (+ beta × KL-Term beim VAE)   Abschnitte 3, 8 und 9, Teile 1 und 2
# ```

# %% [markdown]
# ## 0. Setup

# %%
try:
    import torch  # torch ist auf Colab und dem Jupyter-Server meist schon installiert
    print(f"PyTorch {torch.__version__} bereits vorhanden, kein Neuinstall noetig.")
except ImportError:
    !pip install -q torch

# matplotlib für die Bilder und Diagramme, tqdm für den Fortschrittsbalken beim Training
!pip install -q matplotlib==3.11.2 tqdm==4.70.1

# %% [markdown]
# ## 1. Ersatzbilder statt FashionMNIST
#
# FashionMNIST müsste erst heruntergeladen werden, und ein Training darauf dauert auf der CPU
# mehrere Minuten pro Durchlauf. Wir erzeugen deshalb kleine Ersatzbilder im selben Format: ein
# Kanal, 28 × 28 Pixel, Grauwerte zwischen 0 und 1, genau so, wie `transforms.ToTensor()` die
# FashionMNIST-Bilder in der Aufgabe liefert. Jedes Ersatzbild zeigt eine von drei Formen, einen
# gefüllten Kreis, den Rahmen eines Quadrats oder einen waagerechten Balken, in zufälliger Lage,
# Größe und Helligkeit. Eine vierte Form, ein Kreuz, kommt im Training nie vor; sie wird in
# Abschnitt 2 gebraucht.
#
# Die Ersatzbilder haben einen Vorteil, den echte Daten nicht haben: Wir wissen genau, wodurch
# sich zwei Bilder unterscheiden (Form, Lage, Größe, Helligkeit). Das hilft später, den latenten
# Raum zu deuten. Die Form dient dabei nur zum Einfärben von Diagrammen. Der Autoencoder selbst
# bekommt sie nie zu sehen, er lernt nur aus den Pixeln.

# %%
import torch

ALL_SHAPES = ["Kreis", "Quadrat", "Balken", "Kreuz"]  # Index in dieser Liste = Nummer der Form
TRAIN_SHAPES = ["Kreis", "Quadrat", "Balken"]  # das Kreuz bleibt außen vor


def make_images(n_images, shapes, seed):
    # Eigener Generator: dieselben Bilder bei jedem Lauf, ohne den globalen Zufall zu verstellen
    generator = torch.Generator().manual_seed(seed)
    # Koordinatengitter: ys[i, j] = i (Zeile), xs[i, j] = j (Spalte)
    ys, xs = torch.meshgrid(torch.arange(28.0), torch.arange(28.0), indexing="ij")
    images = torch.zeros(n_images, 1, 28, 28)
    shape_ids = torch.zeros(n_images, dtype=torch.long)
    for label in range(n_images):
        shape_name = shapes[torch.randint(len(shapes), (1,), generator=generator).item()]
        center_x, center_y = (14 + torch.empty(2).uniform_(-5, 5, generator=generator)).tolist()  # Mittelpunkt
        radius = torch.empty(1).uniform_(4, 8, generator=generator).item()  # Größe
        brightness = torch.empty(1).uniform_(0.6, 1.0, generator=generator).item()  # Helligkeit der Form
        # dist = Abstand jedes Pixels zum Rand der Form, innen negativ, außen positiv
        if shape_name == "Kreis":
            dist = ((xs - center_x) ** 2 + (ys - center_y) ** 2).sqrt() - radius
        elif shape_name == "Quadrat":
            dist = (torch.maximum((xs - center_x).abs(), (ys - center_y).abs()) - radius).abs() - 1.0  # nur der Rahmen
        elif shape_name == "Balken":
            dist = torch.maximum((xs - center_x).abs() - 1.4 * radius, (ys - center_y).abs() - 1.5)
        else:  # Kreuz: ein waagerechter und ein senkrechter Balken
            horizontal = torch.maximum((xs - center_x).abs() - radius, (ys - center_y).abs() - 1.2)
            vertical = torch.maximum((xs - center_x).abs() - 1.2, (ys - center_y).abs() - radius)
            dist = torch.minimum(horizontal, vertical)
        # Innen etwa hell, außen etwa 0, am Rand ein weicher Übergang über gut ein Pixel
        images[label, 0] = brightness * torch.sigmoid(-2 * dist)
        shape_ids[label] = ALL_SHAPES.index(shape_name)
    return images, shape_ids


train_images, train_shapes = make_images(3000, TRAIN_SHAPES, seed=1)
test_images, test_shapes = make_images(1000, TRAIN_SHAPES, seed=2)
cross_images, _ = make_images(300, ["Kreuz"], seed=3)  # nur für Abschnitt 2

print(f"train_images: {tuple(train_images.shape)} = 3000 Bilder, 1 Kanal, 28 × 28 Pixel")
print(f"Wertebereich: {train_images.min():.2f} bis {train_images.max():.2f}")

# %% [markdown]
# Die Form (Anzahl, 1, 28, 28) und der Wertebereich 0 bis 1 entsprechen einem Batch aus dem
# `train_loader` der Aufgabe. Je ein Beispiel jeder Form, dazu das Kreuz:

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 4, figsize=(8, 2.3))
for shape_id, ax in enumerate(axes[:3]):
    index = (train_shapes == shape_id).nonzero()[0].item()  # erstes Trainingsbild dieser Form
    ax.imshow(train_images[index, 0], cmap="gray", vmin=0, vmax=1)
    ax.set_title(ALL_SHAPES[shape_id])
axes[3].imshow(cross_images[0, 0], cmap="gray", vmin=0, vmax=1)
axes[3].set_title("Kreuz (nur Test)")
for ax in axes:
    ax.axis("off")
plt.show()

# %% [markdown]
# Die Formen sind bewusst einfach. Ein gefüllter Kreis und ein Quadratrahmen unterscheiden sich
# trotzdem deutlich in ihrer Pixelstruktur, und Lage und Größe ändern sich von Bild zu Bild. Wie
# in der Aufgabe packen wir die Bilder in einen `DataLoader`. Jeder Batch liefert ein Paar aus
# Bildern und einer zweiten Größe; dort ist es die Klasse des Kleidungsstücks, hier die Nummer
# der Form. Die Trainingsschleife ignoriert sie in beiden Fällen. Die Aufgabe nennt diese Größe
# `i`; hier heißt sie `label`, weil `i` üblicherweise für einen Zähler steht.

# %%
from torch.utils.data import DataLoader, TensorDataset

batch_size = 64
# shuffle mit festem Generator, damit die Batches bei jedem Lauf gleich gemischt werden
train_loader = DataLoader(TensorDataset(train_images, train_shapes), batch_size=batch_size,
                          shuffle=True, generator=torch.Generator().manual_seed(0))
test_loader = DataLoader(TensorDataset(test_images, test_shapes), batch_size=batch_size)

image, label = next(iter(train_loader))  # in der Aufgabe heißt label dort i
print(f"image: {tuple(image.shape)} = {batch_size} Bilder, 1 Kanal, 28 × 28 Pixel")
print(f"label: {tuple(label.shape)} = je Bild die Nummer der Form, z. B. {label[:5].tolist()}")

# %% [markdown]
# ## 2. Der Engpass
#
# Mit den Ersatzbildern aus Abschnitt 1 lässt sich jetzt die Frage klären, die in der Aufgabe
# hinter `latent_size` steht. Ein Autoencoder soll seine Eingabe wiederherstellen. Das kann er am
# einfachsten, wenn er sie unverändert durchreicht. Das Buch weist deshalb gleich zu Beginn des
# Kapitels auf den Engpass (Bottleneck) hin: Die mittlere Schicht muss deutlich kleiner sein als
# die Eingabe, sonst lernt das Netz nur, die Eingabewerte zu kopieren, und erfasst dabei nichts
# über die Struktur der Daten. In den Folien ist das die schmale Schicht in der Mitte zwischen
# Encoder und Decoder.
#
# Wir prüfen das mit drei Autoencodern, die sich nur in der Größe des latenten Raums
# unterscheiden. Die beiden Netze mit Engpass bestehen aus zwei `Linear`-Schichten mit ReLU im
# Encoder und zwei im Decoder; das ist die einfachere der beiden Architekturen, die die Aufgabe
# vorschlägt. Warum am Ende Sigmoid steht, klärt Abschnitt 3.

# %%
import torch.nn as nn


class SmallAutoencoder(nn.Module):
    def __init__(self, latent_size, hidden_size=256):
        super().__init__()
        if hidden_size is None:  # ohne Zwischenschicht: direkt vom Bild in den latenten Raum
            encoder_layers = [nn.Linear(784, latent_size)]
            decoder_layers = [nn.Linear(latent_size, 784)]
        else:
            encoder_layers = [nn.Linear(784, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, latent_size)]  # hier liegt der Engpass
            decoder_layers = [nn.Linear(latent_size, hidden_size), nn.ReLU(),
                                 nn.Linear(hidden_size, 784)]
        # Flatten: (Batch, 1, 28, 28) -> (Batch, 784), weil Linear einen Vektor je Bild braucht
        self.encoder = nn.Sequential(nn.Flatten(), *encoder_layers)
        # Unflatten: (Batch, 784) -> (Batch, 1, 28, 28), damit wieder ein Bild herauskommt
        self.decoder = nn.Sequential(*decoder_layers, nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28)))

    def forward(self, x):
        return self.decoder(self.encoder(x))


# %% [markdown]
# Mit `hidden_size=256` wäre die Zwischenschicht selbst schon ein Engpass. Für den Vergleich ohne
# Engpass lassen wir sie deshalb weg (`hidden_size=None`) und machen den latenten Raum so groß
# wie das Bild: 784 Zahlen hinein, 784 Zahlen in der Mitte, 784 Zahlen heraus.
#
# Trainiert wird mit derselben Schleife wie in der Aufgabe: `MSELoss` vergleicht jedes Pixel der
# Rekonstruktion mit demselben Pixel im Eingabebild, das Eingabebild ist also zugleich das Ziel.
# Die Schleife steckt in einer Funktion, damit wir sie mehrfach aufrufen können. Neu gegenüber
# der Aufgabe sind nur `model.train()` und `model.eval()`; warum sie dort stehen, zeigt Abschnitt
# 5.

# %%
from tqdm import tqdm  

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_autoencoder(model, number_epochs=30, learning_rate=1e-3):
    model = model.to(device)
    criterion = nn.MSELoss()  # mittlerer quadratischer Fehler über alle Pixel des Batches
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_loss, test_loss = [], []
    with tqdm(range(number_epochs)) as iterator:
        for epoch in iterator:
            model.train()
            running_loss, count = 0, 0
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.to(device)
                output = model(image)
                loss = criterion(output, image)  # Ziel ist das Eingabebild selbst
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                count += 1
            train_loss.append(running_loss / count)

            model.eval()
            with torch.no_grad():
                running_loss, count = 0, 0
                for image, label in test_loader:
                    image = image.to(device)
                    running_loss += criterion(model(image), image).item()
                    count += 1
                test_loss.append(running_loss / count)
            iterator.set_postfix_str(f"Train Loss: {train_loss[-1]:.4f}, Test Loss: {test_loss[-1]:.4f}")
    return train_loss, test_loss


# Hier ändern: welche Größen des latenten Raums verglichen werden; None = ohne Engpass
latent_sizes = [2, 8, None]

models, curves = {}, {}
for latent_size in latent_sizes:
    torch.manual_seed(0)  # gleiche Startgewichte bei jedem Lauf
    if latent_size is None:
        model = SmallAutoencoder(latent_size=784, hidden_size=None)
    else:
        model = SmallAutoencoder(latent_size)
    models[latent_size] = model
    curves[latent_size] = train_autoencoder(model)

for latent_size in latent_sizes:
    name = "ohne Engpass" if latent_size is None else f"latent_size = {latent_size}"
    print(f"{name:<16} Test-Loss nach 30 Epochen: {curves[latent_size][1][-1]:.4f}")


# %% [markdown]
# Je kleiner der latente Raum, desto größer der Test-Loss: Mit 2 Zahlen pro Bild liegt der
# mittlere quadratische Fehler bei etwa 0,015, mit 8 Zahlen bei etwa 0,002 und ohne Engpass noch
# einmal darunter. Nach diesen Zahlen allein wäre das Netz ohne Engpass das beste. Was die drei
# Netze tatsächlich gelernt haben, zeigt sich aber erst an Bildern, die sie im Training nie
# gesehen haben, den Kreuzen:

# %%
def show_reconstructions(inputs, panel_title):
    rows = [("Eingabe", inputs)]
    with torch.no_grad():
        for latent_size in latent_sizes:
            name = "ohne Engpass" if latent_size is None else f"latent {latent_size}"
            models[latent_size].eval()
            rows.append((name, models[latent_size](inputs.to(device)).cpu()))
    fig, axes = plt.subplots(len(rows), len(inputs), figsize=(1.4 * len(inputs), 1.5 * len(rows)))
    for row, (name, images) in enumerate(rows):
        for column in range(len(inputs)):
            axes[row, column].imshow(images[column, 0], cmap="gray", vmin=0, vmax=1)
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
        axes[row, 0].set_ylabel(name)
    fig.suptitle(panel_title)
    plt.show()


# Je vier Testbilder der bekannten Formen und vier Kreuze nebeneinander
show_reconstructions(torch.cat([test_images[:4], cross_images[:4]]),
                       "links: bekannte Formen, rechts: Kreuze (im Training nie gesehen)")

# %% [markdown]
# Die oberste Zeile zeigt die Eingaben, darunter die Rekonstruktionen der drei Netze. Bei den
# bekannten Formen links wird das Bild mit wachsendem latentem Raum schärfer; mit nur 2 Zahlen
# sind die Ränder verwaschen. Rechts wird der Unterschied grundsätzlich:
#
# - Mit `latent_size = 2` wird aus einem Kreuz ein Balken oder ein Fleck. Das Netz kann nur
#   ausdrücken, was es aus den Trainingsbildern gelernt hat, und ein Kreuz gehört nicht dazu.
# - Mit `latent_size = 8` wird aus jedem Kreuz ein heller Fleck, ähnlich einem kleinen Kreis.
#   Auch hier wird das Kreuz zu einer der bekannten Formen hingezogen.
# - Ohne Engpass kommt das Kreuz erkennbar wieder heraus, wenn auch mit blasseren Armen. Das Netz
#   hat weitgehend gelernt, Pixel durchzureichen, egal was das Bild zeigt.
#
# Das ist der Punkt aus dem Buch: Ohne Engpass entsteht eine gute Kopiermaschine, aber keine
# verdichtete Beschreibung der Daten. Das Kreuz-Experiment zeigt zugleich die Idee hinter der
# Anomalieerkennung aus den Folien: Ein Netz mit Engpass rekonstruiert untypische Eingaben
# schlecht, und genau daran lassen sie sich erkennen.
#
# Für die Wahl von `latent_size` in der Aufgabe heißt das: Der Wert muss klein genug sein, dass
# ein echter Engpass entsteht, und groß genug, dass die Rekonstruktionen nicht verschwimmen.
# FashionMNIST ist deutlich vielfältiger als unsere drei Formen; dort sieht man am Rand von
# Schuhen und an den Mustern von Hemden schnell, ob der latente Raum zu klein ist. Speichern Sie
# dazu wie in der Aufgabe vorgesehen die Bilder `img_*.png` und `rec_*.png` und vergleichen Sie
# sie.

# %% [markdown]
# ## 3. Letzte Aktivierung, Wertebereich und Verlustfunktion
#
# Der Decoder in Abschnitt 2 endete mit einer Sigmoid-Funktion. Diese Entscheidung verlangt auch
# die Aufgabe von Ihnen: Welche Aktivierung am Ende Grauwerte zwischen 0 und 1 erzeugt. Die
# Antwort hängt davon ab, welche Werte die verschiedenen Aktivierungen überhaupt ausgeben können.
# Wir schicken dazu Werte von −3 bis 3 durch die drei Funktionen, die bisher im Kurs vorkamen:

# %%
values = torch.linspace(-3, 3, 7)  # -3, -2, ..., 3
print("Eingabe: ", [f"{w:5.2f}" for w in values.tolist()])
print("ReLU:    ", [f"{w:5.2f}" for w in torch.relu(values).tolist()])
print("Tanh:    ", [f"{w:5.2f}" for w in torch.tanh(values).tolist()])
print("Sigmoid: ", [f"{w:5.2f}" for w in torch.sigmoid(values).tolist()])

# %% [markdown]
# ReLU schneidet nur nach unten bei 0 ab und ist nach oben offen. Tanh liegt immer zwischen −1
# und 1, Sigmoid immer zwischen 0 und 1; beide erreichen ihre Grenzen nie ganz, kommen ihnen bei
# großen Beträgen aber sehr nahe. Alle drei wirken auf jeden Wert einzeln. Softmax aus der
# Klassifikation gehört nicht in diese Reihe: Sie verteilt eine Summe von 1 auf alle Ausgaben
# gemeinsam, die Grauwerte eines Bildes müssen sich aber nicht zu 1 aufaddieren.
#
# Das Buch verwendet in seinen Autoencoder-Beispielen `Tanh`. Das passt dort, weil die Bilder
# vorher mit `transforms.Normalize((0.5, ...), (0.5, ...))` von 0 bis 1 auf −1 bis 1 verschoben
# werden: Von jedem Wert wird 0,5 abgezogen, dann wird durch 0,5 geteilt. Die Hilfsfunktion
# `to_image` im Buch schiebt die Werte vor dem Speichern mit `0.5 * (x + 1)` wieder zurück. In
# der Aufgabe fehlt beides: Die Bilder kommen mit `ToTensor()` allein, also mit Werten von 0 bis
# 1, und `to_image` schneidet nur ab. Was das für eine Ausgabe bedeutet, die nicht im passenden
# Bereich liegt, zeigt ein kleines Beispiel:

# %%
example_outputs = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0, 1.4])  # denkbare Ausgaben eines Decoders


def to_image_book(x):
    return (0.5 * (x + 1)).clamp(0, 1)  # Buch: von -1..1 zurück auf 0..1, dann abschneiden


def to_image_task(x):
    return x.clamp(0, 1)  # Aufgabe: nur abschneiden (ohne das view auf 28 × 28)


print("Ausgabe:          ", example_outputs.tolist())
print("to_image (Buch):  ", to_image_book(example_outputs).tolist())
print("to_image (Aufgabe):", to_image_task(example_outputs).tolist())

# %% [markdown]
# Mit der Version der Aufgabe werden alle negativen Werte zu 0 und alle Werte über 1 zu 1. In den
# gespeicherten Bildern sieht man deshalb nicht, ob der Decoder Werte außerhalb von 0 bis 1
# ausgibt. Die Verlustfunktion sieht es sehr wohl, denn sie vergleicht die unbeschnittene Ausgabe
# mit dem Ziel. Ein Decoder, dessen letzte Aktivierung gar nicht in den Bereich der Zielwerte
# passt, verschenkt Trainingsaufwand oder kann bestimmte Grauwerte nie treffen; die gespeicherten
# Bilder verraten das nicht.
#
# Bei der Wahl von `criterion` spielt der Wertebereich ebenfalls eine Rolle. `MSELoss`
# funktioniert mit beliebigen Zahlen. `BCELoss`, die binäre Kreuzentropie, ist dagegen für
# Wahrscheinlichkeiten gebaut und verlangt Ausgaben zwischen 0 und 1; die Grauwerte des Ziels
# werden dann wie Wahrscheinlichkeiten behandelt. Das Buch verwendet sie im VAE-Beispiel (`BCE`
# in `loss_function`). Liegt eine Ausgabe außerhalb, bricht sie mit einem Fehler ab:

# %%
target = torch.tensor([0.0, 0.3, 1.0])
print("MSELoss:", nn.MSELoss()(torch.tensor([0.1, 0.4, 1.4]), target).item())
try:
    nn.BCELoss()(torch.tensor([0.1, 0.4, 1.4]), target)
except RuntimeError as error:
    print("BCELoss:", error)

# %% [markdown]
# `MSELoss` rechnet mit dem Wert 1,4 einfach weiter, `BCELoss` lehnt ihn ab. Welche letzte
# Aktivierung zu Zielwerten zwischen 0 und 1 passt und beide Verlustfunktionen zulässt, lässt
# sich an der ersten Ausgabe dieses Abschnitts ablesen.

# %% [markdown]
# ## 4. Formen durch Conv2d und ConvTranspose2d
#
# Die Netze in den Abschnitten 2 und 3 haben jedes Bild mit `Flatten` zu einem Vektor aus 784
# Zahlen gemacht. Dabei geht verloren, welche Pixel nebeneinander liegen, wie beim MLP aus Termin
# 1. Die zweite Architektur der Aufgabe nutzt deshalb Faltungen wie die Bildklassifikation aus
# Termin 2: Der Encoder verkleinert das Bild Schritt für Schritt und erhöht dabei die Zahl der
# Kanäle, der Decoder geht denselben Weg zurück. Das Buch beschreibt den Convolutional
# Autoencoder genau so: Durch das Verkleinern des Bildes muss der Encoder eine komprimierte
# Version lernen.
#
# Für den Decoder schlägt die Aufgabe `ConvTranspose2d` vor. Wie die Bildgröße sich durch die
# beiden Schichten ändert, sehen wir zuerst an einem einzelnen Block ohne Padding und mit
# Schrittweite 1:

# %%
x = torch.rand(1, 1, 28, 28)  # ein Bild: (Batch, Kanäle, Höhe, Breite)
conv = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5)
deconv = nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=5)

h = conv(x)
print(f"nach Conv2d:          {tuple(h.shape)} = 1 Bild, 8 Kanäle, 24 × 24")
print(f"nach ConvTranspose2d: {tuple(deconv(h).shape)} = wieder 1 Kanal, 28 × 28")

# %% [markdown]
# Ein 5 × 5-Filter passt in einer Zeile mit 28 Pixeln an 24 Stellen; ab der 25. Stelle würde er
# über den Rand ragen. Ohne Padding verliert das Bild also bei Kernelgröße 5 in Breite und Höhe
# je vier Pixel, zwei an jedem Rand; bei Kernelgröße 3 sind es je zwei. `ConvTranspose2d` mit
# denselben Einstellungen macht genau das rückgängig: Jedes Eingabepixel verteilt sich auf ein
# 5 × 5-Feld, und aus 24 Positionen werden wieder 28. Wichtig: Es stellt nur die Form wieder
# her, nicht die Werte. `ConvTranspose2d` ist keine Umkehrfunktion von `Conv2d`, sondern eine
# eigene Schicht mit eigenen Gewichten, die das Training erst lernen muss.
#
# Mit Schrittweite 2 halbiert sich die Bildgröße. Hier gibt es einen Haken:

# %%
conv_stride2 = nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
print(f"28 × 28 nach Conv2d mit stride=2: {tuple(conv_stride2(torch.rand(1, 1, 28, 28)).shape)}")
print(f"27 × 27 nach Conv2d mit stride=2: {tuple(conv_stride2(torch.rand(1, 1, 27, 27)).shape)}")

h = conv_stride2(x)
deconv_without = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1)
deconv_with = nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1, output_padding=1)
print(f"zurück ohne output_padding: {tuple(deconv_without(h).shape)}")
print(f"zurück mit output_padding=1: {tuple(deconv_with(h).shape)}")

# %% [markdown]
# Ein 28er- und ein 27er-Bild landen beide bei 14 × 14. Der Rückweg kann daher nicht wissen,
# welche der beiden Größen gemeint war, und wählt ohne weitere Angabe die kleinere, 27.
# `output_padding=1` hängt die fehlende Zeile und Spalte an. Bei Schrittweite 1 tritt das Problem
# nicht auf.
#
# Wie mehrere solcher Schritte zusammenwirken, zeigt das Conv-Beispiel im Buch. Wir schicken ein
# Bild durch dessen Encoder und Decoder und geben nach jeder Schicht die Form aus:

# %%
# Encoder und Decoder aus dem Convolutional-Autoencoder-Beispiel im Buch, Schichten unverändert.
# Kommentar je Schicht: was sie tut, Form danach als Kanäle × Höhe × Breite
book_encoder = nn.Sequential(
    nn.Conv2d(1, 16, 3, stride=3, padding=1),  # 16 Filter suchen Muster in 3×3-Kacheln -> 16×10×10
    nn.ReLU(True),  # Nichtlinearität, negative Werte -> 0 (True: direkt im Tensor, spart Speicher)
    nn.MaxPool2d(2, stride=2),  # je 2×2-Block bleibt der stärkste Wert -> 16×5×5
    nn.Conv2d(16, 8, 3, stride=2, padding=1),  # verdichtet 16 Musterkanäle zu 8 -> 8×3×3
    nn.ReLU(True),
    nn.MaxPool2d(2, stride=1),  # -> 8×2×2 = 32 Zahlen: der Engpass
)
book_decoder = nn.Sequential(
    nn.ConvTranspose2d(8, 16, 3, stride=2),  # breitet jedes Pixel auf 3×3 aus -> 16×5×5
    nn.ReLU(True),
    nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # vergrößert weiter -> 8×15×15
    nn.ReLU(True),
    nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # ein Graukanal in Bildgröße -> 1×28×28
    nn.Tanh(),  # Form bleibt, Werte werden auf -1 bis 1 gebracht wie die Zielbilder
)

h = x
for layer in list(book_encoder) + list(book_decoder):
    h = layer(h)
    if not isinstance(layer, nn.ReLU):  # ReLU ändert die Form nicht
        print(f"{type(layer).__name__:<16} -> {tuple(h.shape)}")

# %% [markdown]
# Der Encoder verkleinert das Bild in vier Schritten auf 8 × 2 × 2 = 32 Zahlen, der Decoder baut
# in drei Schritten wieder 28 × 28 auf. Er muss nicht spiegelbildlich zum Encoder sein, solange
# am Ende die richtige Größe herauskommt. Die erste Faltung springt mit Schrittweite 3 genau um
# ihre Kernelgröße weiter; sie zerlegt das um einen Rand erweiterte 30er-Bild also in 10 × 10
# Kacheln, die sich nicht überlappen.
#
# Warum wechseln die Kanäle von 1 über 16 auf 8? Die Pixel einer Schicht sagen, wo etwas ist, die
# Kanäle, was dort ist: Jeder Kanal gehört zu einem eigenen Filter, der ein eigenes Muster
# erkennt, etwa waagerechte Kanten oder helle Flecken. Die 1 am Eingang ist vorgegeben, weil das
# Bild nur einen Graukanal hat. Bliebe es bei einem Kanal, könnte die erste Schicht nur ein
# einziges Muster erkennen, und ein waagerechter und ein senkrechter Strich wären danach nicht
# mehr zu unterscheiden. Mehr Pixel, etwa 40 × 40 statt 10 × 10, würden daran nichts ändern;
# derselbe Filter würde nur an mehr Stellen abgefragt. Die erste Faltung tauscht deshalb Ort
# gegen Art: Sie verkleinert das Bild, erzeugt dafür aber 16 Kanäle. Erst danach wird verdichtet:
#
# ```
# Eingabe        1 × 28 × 28     784 Zahlen
# 1. Faltung    16 × 10 × 10   1.600
# 1. Pooling    16 ×  5 ×  5     400
# 2. Faltung     8 ×  3 ×  3      72
# 2. Pooling     8 ×  2 ×  2      32   Engpass
# ```
#
# Die Senkung auf 8 Kanäle macht den Engpass klein; mit 16 Kanälen blieben am Ende 64 Zahlen. Die
# genauen Werte 16 und 8 sind frei gewählt, wie `hidden_size` in Abschnitt 2.
#
# Das abschließende `Tanh` ändert die Form nicht mehr, nur die Werte. Eine
# `ConvTranspose2d`-Schicht kann beliebige Zahlen ausgeben; Tanh drückt sie in den Bereich −1 bis
# 1, in dem die Zielbilder des Buchs nach `Normalize` liegen (Abschnitt 3). Die Faltungen sorgen
# also für die Form, die letzte Aktivierung für den Wertebereich. Ein Durchlauf mit einem
# Zufallsbild wie hier ist die schnellste Art, die Form zu prüfen, bevor man trainiert.
#
# Die Aufgabe hängt an den Encoder zusätzlich eine `Linear`-Schicht, die in den latenten Raum mit
# `latent_size` Zahlen führt. Dort bildet diese Schicht den Engpass, deshalb dürfen die Kanäle in
# den Faltungen davor durchgehend steigen, statt wie im Buch wieder zu sinken. Dazwischen steht
# `Flatten`, im Decoder auf dem Rückweg `Unflatten`. Die Zahl, die dafür in `nn.Linear` gehört,
# ergibt sich aus der Form nach der letzten Faltung. Am Beispiel des einzelnen Blocks von oben (8
# Kanäle, 24 × 24):

# %%
batch = torch.rand(64, 1, 28, 28)
h = conv(batch)  # (64, 8, 24, 24)
flat = nn.Flatten()(h)  # alles außer der Batch-Achse hintereinander
print(f"nach Flatten:   {tuple(flat.shape)} = 64 Bilder, je 8 × 24 × 24 = {8 * 24 * 24} Zahlen")

to_latent = nn.Linear(8 * 24 * 24, 2)  # 4608 Zahlen -> 2 Zahlen
from_latent = nn.Linear(2, 8 * 24 * 24)  # Rückweg im Decoder
z = to_latent(flat)
print(f"latent:         {tuple(z.shape)}")
restored = nn.Unflatten(1, (8, 24, 24))(from_latent(z))  # Achse 1 wieder in Kanäle × Höhe × Breite
print(f"nach Unflatten: {tuple(restored.shape)}, bereit für ConvTranspose2d")

# %% [markdown]
# `nn.Unflatten(1, (8, 24, 24))` ist das Gegenstück zu `Flatten`: Achse 1 mit 4608 Einträgen wird
# wieder in 8 Kanäle mit je 24 × 24 Pixeln aufgeteilt. Beachten Sie die Größe der
# `Linear`-Schicht: Bei 4608 Eingängen und einer Zwischenschicht wie in Abschnitt 2 kommen
# schnell Millionen Gewichte zusammen; jede weitere Faltung, die das Bild verkleinert, macht
# diese Schicht kleiner.
#
# `BatchNorm2d`, das die Aufgabe zwischen Faltung und ReLU vorschlägt, ändert die Form nicht. Es
# normiert jeden Kanal anhand der Werte im aktuellen Batch und stabilisiert so das Training. Dass
# es sich dabei im Training anders verhält als bei der Auswertung, wird im nächsten Abschnitt
# wichtig.

# %% [markdown]
# ## 5. Training und latenter Raum
#
# Abschnitt 4 hat gezeigt, wie die Bausteine eines Convolutional Autoencoders in der Form
# zusammenpassen. Trainiert haben wir bisher nur die einfachen Netze aus Abschnitt 2, und dort
# nur auf den letzten Test-Loss und einige Rekonstruktionen geschaut. Wie in der Aufgabe lohnt
# sich auch der Blick auf die Kurven über alle Epochen:

# %%
fig, axes = plt.subplots(1, len(latent_sizes), figsize=(4 * len(latent_sizes), 3), sharey=True)
for ax, latent_size in zip(axes, latent_sizes):
    train_loss, test_loss = curves[latent_size]
    ax.plot(range(len(train_loss)), train_loss, label="train_loss")
    ax.plot(range(len(test_loss)), test_loss, label="test_loss")
    ax.set(title="ohne Engpass" if latent_size is None else f"latent_size = {latent_size}",
           xlabel="epoch", yscale="log")
axes[0].set_ylabel("loss (logarithmisch)")
axes[0].legend()
plt.show()

# %% [markdown]
# Die y-Achse ist logarithmisch, damit alle drei Netze vergleichbar bleiben. In allen drei Fällen
# fallen Trainings- und Testkurve fast deckungsgleich ab. Das heißt: Keines der Netze lernt die
# Trainingsbilder auswendig, auch das ohne Engpass nicht. Sein Problem aus Abschnitt 2 ist ein
# anderes; es lernt eine allgemeine Kopierfunktion, und die funktioniert auf neuen Bildern
# genauso gut. An den Kurven allein lässt sich ein fehlender Engpass also nicht erkennen, dafür
# braucht es den Blick auf die Bilder.
#
# Die Trainingsschleife in `train_autoencoder` ruft vor dem Training `model.train()` und vor der
# Auswertung `model.eval()` auf, wie in Termin 2. Bei unseren Netzen ohne BatchNorm und ohne
# Dropout ändert das nichts. Sobald Sie BatchNorm verwenden, wie es die CNN-Variante der Aufgabe
# vorschlägt, ist es wichtig. Im Trainingsmodus normiert BatchNorm mit Mittelwert und Streuung
# des aktuellen Batches und aktualisiert nebenbei seine gespeicherten Werte für später; im
# Auswertungsmodus verwendet es nur diese gespeicherten Werte:

# %%
torch.manual_seed(0)
bn = nn.BatchNorm2d(1)
one_image = test_images[:1]  # ein festes Bild
batch_a = torch.cat([one_image, test_images[1:8]])  # dasselbe Bild in zwei verschiedenen Batches
batch_b = torch.cat([one_image, test_images[8:15]])

bn.train()
with torch.no_grad():  # no_grad verhindert nur Gradienten, nicht das Aktualisieren der Statistik
    before = bn.running_mean.item()
    out_a, out_b = bn(batch_a)[0], bn(batch_b)[0]
print(f"train(): gleiches Bild, Mittelwert der Ausgabe {out_a.mean():.3f} gegen {out_b.mean():.3f}")
print(f"         running_mean vorher {before:.4f}, nachher {bn.running_mean.item():.4f}")

bn.eval()
with torch.no_grad():
    out_a, out_b = bn(batch_a)[0], bn(batch_b)[0]
print(f"eval():  gleiches Bild, Mittelwert der Ausgabe {out_a.mean():.3f} gegen {out_b.mean():.3f}")

# %% [markdown]
# Im Trainingsmodus hängt die Ausgabe für dasselbe Bild davon ab, mit welchen anderen Bildern es
# im Batch steht, und schon die beiden Durchläufe verändern `running_mean`, obwohl
# `torch.no_grad()` aktiv ist. Im Auswertungsmodus ist die Ausgabe für dasselbe Bild gleich.
# Fehlt `model.eval()` vor der Testschleife, wird der Test-Loss also mit Batch-Statistiken
# berechnet, und die Testbilder verändern nebenbei die gespeicherten Werte des Modells.
#
# Zurück zu den Netzen aus Abschnitt 2. Das Netz mit `latent_size = 2` hat einen Vorteil: Seinen
# latenten Raum kann man direkt zeichnen. Jedes Testbild wird zu einem Punkt mit zwei
# Koordinaten. Die Folien zeigen so ein Bild für handgeschriebene Ziffern unter "Autoencoder zur
# Dimensionsreduktion". Wir zeichnen die Punkte für ein untrainiertes und das trainierte Netz und
# färben sie nach der Form ein, die das Netz selbst nie gesehen hat:

# %%
torch.manual_seed(0)
untrained = SmallAutoencoder(2).to(device).eval()  # gleiche Startgewichte wie in Abschnitt 2
trained = models[2].eval()

with torch.no_grad():
    codes_before = untrained.encoder(test_images.to(device)).cpu()  # (1000, 2)
    codes = trained.encoder(test_images.to(device)).cpu()

colors = ["tab:blue", "tab:orange", "tab:green"]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, points, panel_title in [(axes[0], codes_before, "untrainiert"), (axes[1], codes, "nach 30 Epochen")]:
    for shape_id in range(3):
        mask = test_shapes == shape_id
        ax.scatter(points[mask, 0], points[mask, 1], s=4, color=colors[shape_id],
                   label=ALL_SHAPES[shape_id])
    ax.set(title=panel_title, xlabel="latente Koordinate 1", ylabel="latente Koordinate 2")
axes[1].legend(markerscale=3)
plt.show()


# %% [markdown]
# Jeder Punkt ist ein Testbild. Beim untrainierten Netz liegen alle Punkte in einem kleinen
# Bereich von wenigen Hundertsteln; Kreise liegen eher rechts, weil auch zufällige Gewichte
# ähnliche Pixelmuster auf ähnliche Punkte abbilden, aber Quadrate und Balken überlappen fast
# vollständig. Nach dem Training ziehen sich die drei Formen zu eigenen Bereichen auseinander,
# die sich nur in der Mitte berühren. Das Netz hat diese Ordnung allein aus der Aufgabe gelernt,
# die Bilder wiederherzustellen. Das ist mit "Dimensionsreduktion" gemeint: 784 Pixel werden zu 2
# Zahlen, die noch das Wesentliche des Bildes beschreiben.
#
# Zugleich sieht man, wie ungleichmäßig der Raum gefüllt ist: Die Bereiche sind unterschiedlich
# groß, die zweite Koordinate reicht viel weiter als die erste, und zwischen den Ausläufern der
# Bereiche bleiben große Flächen leer. Was das für das Erzeugen neuer Bilder bedeutet, untersucht
# der nächste Abschnitt.

# %% [markdown]
# ## 6. Zufällige Punkte im latenten Raum
#
# Wenn jeder Punkt im latenten Raum für ein Bild steht, liegt die Idee "Autoencoder als
# Generator" nahe: Man wählt einen Punkt, gibt ihn dem Decoder und bekommt ein neues Bild, das so
# in den Trainingsdaten nicht vorkam. Wir probieren das mit dem trainierten Netz aus Abschnitt 5
# und legen ein regelmäßiges Gitter aus 8 × 8 Punkten über den Bereich, den die Testbilder
# einnehmen:

# %%
def decode_grid(decoder, x_range, y_range, n=8):
    xs = torch.linspace(*x_range, n)
    ys = torch.linspace(*y_range, n)
    # Zeile r, Spalte c des Gitters = Punkt (xs[c], ys[n-1-r]), damit oben große y-Werte stehen
    points = torch.stack([torch.stack([xs[c], ys[n - 1 - r]]) for r in range(n) for c in range(n)])
    with torch.no_grad():
        images = decoder(points.to(device)).cpu()  # (n*n, 1, 28, 28)
    # n × n Bilder zu einem großen Bild zusammensetzen: (n*28, n*28)
    return images.view(n, n, 28, 28).permute(0, 2, 1, 3).reshape(n * 28, n * 28)


x_range = (codes[:, 0].min().item(), codes[:, 0].max().item())
y_range = (codes[:, 1].min().item(), codes[:, 1].max().item())
print(f"Bereich der Testbilder: Koordinate 1 von {x_range[0]:.1f} bis {x_range[1]:.1f}, "
      f"Koordinate 2 von {y_range[0]:.1f} bis {y_range[1]:.1f}")

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for shape_id in range(3):
    mask = test_shapes == shape_id
    axes[0].scatter(codes[mask, 0], codes[mask, 1], s=4, color=colors[shape_id], label=ALL_SHAPES[shape_id])
grid_x, grid_y = torch.meshgrid(torch.linspace(*x_range, 8), torch.linspace(*y_range, 8), indexing="ij")
axes[0].scatter(grid_x.flatten(), grid_y.flatten(), marker="x", color="black", s=20, label="Gitterpunkte")
axes[0].set(title="Testbilder und Gitterpunkte", xlabel="latente Koordinate 1", ylabel="latente Koordinate 2")
axes[0].legend(markerscale=2, fontsize=8)
axes[1].imshow(decode_grid(trained.decoder, x_range, y_range), cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Decoder-Ausgabe je Gitterpunkt")
axes[1].axis("off")
plt.show()


# %% [markdown]
# Links liegen die Gitterpunkte als Kreuze über den Testbildern, rechts steht an derselben
# Stelle, was der Decoder aus dem jeweiligen Punkt macht. Die Anordnung entspricht dem
# Streudiagramm: Quadrate oben links, Balken unten links, Kreise rechts. Wo Gitterpunkte in einem
# dicht besetzten Bereich liegen, entstehen erkennbare Formen. In den Übergangszonen zwischen den
# Bereichen kommen dagegen blasse Mischformen heraus, halb Balken, halb Fleck, und in der
# obersten Reihe, wo kaum ein Testbild liegt, verzerrte Rahmen. Für diese Punkte hat den Decoder
# im Training nie ein Verlust korrigiert; was er dort ausgibt, ist durch nichts festgelegt.
#
# Dazu kommt ein zweites Problem: Um einen Punkt zufällig zu wählen, muss man wissen, wo die
# Bilder liegen. Hier reicht die erste Koordinate etwa von −10 bis 13, die zweite von −30 bis 5,
# und nach einem neuen Training wären es ganz andere Bereiche. Wählte man die Punkte einfach nach
# der Standardnormalverteilung, landeten fast alle im kleinen Bereich um den Ursprung, wo sich
# die drei Formen überlagern. Das Buch beschreibt diese Grenze des normalen Autoencoders so: Er
# ist vor allem dann nützlich, wenn man die Eingabe wiederherstellen will. Für das Erzeugen neuer
# Daten braucht es einen latenten Raum, der lückenlos gefüllt ist und dessen Lage man vorher
# kennt. Genau das leistet der Variational Autoencoder im zweiten Teil der Aufgabe.

# %% [markdown]
# ## 7. VAE: zwei Ausgaben und `reparametrize`
#
# Abschnitt 6 endete mit der Forderung nach einem latenten Raum ohne Lücken. Der VAE setzt dafür
# beim Codieren an. Das Buch erklärt den Unterschied an Gesichtern: Ein normaler Autoencoder
# würde eine Eigenschaft wie "lächelt" als einen festen Wert codieren, ein VAE dagegen als
# Wahrscheinlichkeitsverteilung über einen Bereich möglicher Werte. Im Ablauf unterscheiden sich
# die beiden Netze nur in der Mitte:
#
# ```
# Autoencoder:  Bild ──▶ Encoder ──▶ z ─────────────────────────────────▶ Decoder ──▶ Bild
#                                    ein Punkt
#
# VAE:          Bild ──▶ Encoder ──▶ mu, logvar ──▶ reparametrize ──▶ z ──▶ Decoder ──▶ Bild
#                                    Glockenkurve:  wählt daraus
#                                    Lage, Breite   zufällig einen Punkt
# ```
#
# Konkret liefert der Encoder für jede latente Dimension zwei Zahlen: den Mittelwert `mu`, wo die
# Glockenkurve liegt, und `logvar`, wie breit sie ist. Der Encoder bekommt dafür zwei getrennte
# `Linear`-Schichten am Ende, eine für jede der beiden Ausgaben. Das VAE-Beispiel im Buch ist so
# aufgebaut (`fc21` und `fc22`), und auch die Klasse `VariationalAutoencoder` der Aufgabe
# erwartet vom Encoder genau dieses Paar:

# %%
class VaeEncoder(nn.Module):
    def __init__(self, latent_size, hidden_size=256):
        super().__init__()
        self.main = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden_size), nn.ReLU())
        self.mu = nn.Linear(hidden_size, latent_size)  # Kopf 1: Mittelwert je Dimension
        self.logvar = nn.Linear(hidden_size, latent_size)  # Kopf 2: log. Varianz je Dimension

    def forward(self, x):
        h = self.main(x)  # gemeinsamer Teil für beide Köpfe
        return self.mu(h), self.logvar(h)


torch.manual_seed(0)
mu, logvar = VaeEncoder(latent_size=2)(test_images[:3])
print(f"mu:     {tuple(mu.shape)} = 3 Bilder, je 2 Mittelwerte")
print(f"logvar: {tuple(logvar.shape)} = 3 Bilder, je 2 logarithmierte Varianzen")

# %% [markdown]
# Warum `logvar` und nicht direkt die Streuung? Eine Streuung muss positiv sein, eine
# `Linear`-Schicht kann aber jede Zahl ausgeben. Den Logarithmus einer Varianz darf man dagegen
# frei wählen: Jede Zahl, auch eine negative, gehört zu einer gültigen positiven Varianz. Gemeint
# ist dabei der Logarithmus der Varianz, also der quadrierten Streuung, nicht der Streuung
# selbst. Das erklärt die erste Zeile der Methode `reparametrize` aus der Aufgabe. Wir gehen sie
# mit drei Beispielwerten durch:

# %%
device_before = device
device = torch.device("cpu")  # reparametrize verwendet die globale Variable device

# reparametrize aus der Klasse VariationalAutoencoder der Aufgabe, Zeile für Zeile
mu = torch.tensor([0.0, 2.0, -1.0])
logvar = torch.tensor([0.0, -2.0, 1.0])

std = logvar.mul(0.5).exp_()  # halber Logarithmus der Varianz, dann e hoch: die Streuung
print("std:", [round(s, 3) for s in std.tolist()])

torch.manual_seed(0)
eps = torch.FloatTensor(std.size()).normal_().to(device)  # Zufallszahlen aus N(0, 1)
print("eps:", [round(e, 3) for e in eps.tolist()])

z = eps.mul(std).add_(mu)  # eps strecken, dann verschieben
print("z:  ", [round(v, 3) for v in z.tolist()])

# %% [markdown]
# Die erste Zeile macht aus `logvar` die Streuung: `logvar = 0` gehört zur Varianz 1 und damit
# zur Streuung 1, `logvar = -2` zur Streuung 0,37, `logvar = 1` zur Streuung 1,65. Die beiden
# anderen Zeilen zeigen, wie `z` entsteht: `reparametrize` erzeugt den Zufallswert nicht direkt
# mit der Verteilung, die der Encoder beschreibt. Stattdessen erzeugt es zuerst eine Zufallszahl
# `eps` aus der Standardnormalverteilung und rechnet sie anschließend mit `std` und `mu` um.
#
# Die folgende Grafik zeigt das für die drei Beispiele, eine Zeile je Beispiel. Die gestrichelte
# graue Kurve ist in jeder Zeile gleich: Sie zeigt, wie die Zufallszahlen `eps` verteilt sind.
# Die meisten liegen nahe bei 0, Werte über 2 oder unter −2 sind selten. Die farbige Kurve zeigt,
# wie `z` nach der Umrechnung verteilt ist. Der graue Punkt ist der Wert von `eps` aus der
# vorigen Zelle, der Pfeil führt zum daraus berechneten `z`. Damit man sieht, dass die farbige
# Kurve stimmt, rechnen wir zusätzlich 10.000 Zufallszahlen auf dieselbe Weise um und zeichnen
# sie als helle Säulen:

# %%
from torch.distributions import Normal


def fmt_de(x):  # Zahl mit deutschem Komma und echtem Minuszeichen
    return f"{x:.2f}".replace(".", ",").replace("-", "−")


many_eps = torch.randn(10000)  # 10.000 Zufallszahlen aus der Standardnormalverteilung
x_values = torch.linspace(-7, 5, 400)  # x-Werte, an denen die Kurven gezeichnet werden
example_colors = ["tab:blue", "tab:orange", "tab:green"]
example_titles = ["Beispiel 1: Skalierung mit std = 1, Verschiebung um mu = 0 (unverändert)",
                  "Beispiel 2: Skalierung mit std = 0,37, Verschiebung um mu = 2",
                  "Beispiel 3: Skalierung mit std = 1,65, Verschiebung um mu = −1"]

fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, sharey=True)
for example, ax in enumerate(axes):  # example = Nummer des Beispiels (0, 1, 2)
    line_color = example_colors[example]
    # Grau: Verteilung von eps, in jeder Zeile gleich
    ax.plot(x_values, Normal(0.0, 1.0).log_prob(x_values).exp(), color="gray", linestyle="--",
            label="Dichte von eps (Standardnormalverteilung)")
    # Hell: 10.000 Zufallszahlen, umgerechnet wie in reparametrize
    ax.hist(many_eps * std[example] + mu[example], bins=100, range=(-7, 5), density=True, color=line_color,
            alpha=0.25, label="10.000 umgerechnete Zufallszahlen")
    # Farbig: Verteilung von z, die der Encoder mit mu und std beschreibt
    ax.plot(x_values, Normal(mu[example], std[example]).log_prob(x_values).exp(), color=line_color,
            label=f"Dichte von z (mu = {fmt_de(mu[example])}, std = {fmt_de(std[example])})")
    # Der eine Wert aus der vorigen Zelle: eps (grauer Punkt) wird zu z (farbiger Punkt)
    ax.plot(eps[example], 0, "o", color="gray", markersize=9, clip_on=False, zorder=5,
            label=f"eps aus der vorigen Zelle: {fmt_de(eps[example])}")
    ax.plot(z[example], 0, "o", color=line_color, markersize=9, clip_on=False, zorder=5,
            label=f"daraus berechnetes z: {fmt_de(z[example])}")
    ax.annotate("", xy=(z[example], 0.1), xytext=(eps[example], 0.1),  # Pfeil von eps nach z
                arrowprops=dict(arrowstyle="->", color=line_color, lw=2))
    ax.set_title(example_titles[example], loc="left")
    ax.set_ylabel("Dichte")
    ax.legend(loc="upper left", fontsize=8)
axes[-1].set_xlabel("Wert")
axes[0].set_ylim(0, 1.15)
plt.tight_layout()
plt.show()

# %% [markdown]
# - Beispiel 1 (`mu = 0`, `logvar = 0`, also `std = 1`): Die Zahl bleibt, wie sie ist. Aus 1,54
#   wird wieder 1,54; deshalb liegen der graue und der blaue Punkt übereinander, und die blaue
#   Kurve verdeckt die graue.
# - Beispiel 2 (`mu = 2`, `logvar = -2`, also `std = 0,37`): Die Zahl wird erst auf gut ein
#   Drittel verkleinert und dann um 2 erhöht. Aus −0,29 wird 1,89. Weil alle Zufallszahlen so
#   behandelt werden, landen sie dicht gedrängt um 2: Die orange Kurve ist schmal und hoch.
# - Beispiel 3 (`mu = −1`, `logvar = 1`, also `std = 1,65`): Die Zahl wird vergrößert und dann um
#   1 verringert. Aus −2,18 wird −4,59. Die Werte verteilen sich dadurch weit um −1: Die grüne
#   Kurve ist flach und breit.
#
# Kurz gesagt: `mu` legt fest, wo die Werte liegen, `std`, wie weit sie streuen. Die hellen
# Säulen der 10.000 umgerechneten Zufallszahlen passen in jeder Zeile genau auf die farbige
# Kurve. Die Umrechnung liefert also genau die Verteilung, die der Encoder mit `mu` und `logvar`
# beschreibt.

# %% [markdown]
# Warum dieser Umweg über `eps`? Beim Training berechnet `loss.backward()` für jeden Parameter
# den Gradienten, also wie stark sich der Verlust ändert, wenn der Parameter ein wenig verändert
# wird (Termin 1). Für `mu` hängt das an einer einfachen Frage: Wie ändert sich `z`, wenn `mu`
# etwas größer wird? Der folgende Versuch vergrößert `mu` von 2,00 auf 2,01 und berechnet `z` auf
# beiden Wegen, die direkte Zufallsstichprobe dreimal hintereinander:

# %%
# Was passiert mit z, wenn mu um 0,01 wächst? Streuung jeweils 1
torch.manual_seed(1)
eps_fixed = torch.randn(1)  # für reparametrize: eine Zufallszahl, die danach fest bleibt
for mu_value in [2.00, 2.01]:
    # Weg 1: dreimal direkt zufällig erzeugen, jedes Mal neu gewürfelt
    direct = [torch.normal(torch.tensor(mu_value), torch.tensor(1.0)).item() for _ in range(3)]
    # Weg 2: dieselbe Rechnung wie in reparametrize, mit festem eps
    reparam = (eps_fixed * 1.0 + mu_value).item()
    print(f"mu = {mu_value:.2f}:  direkt erzeugt {[round(v, 2) for v in direct]},"
          f"  reparametrize {reparam:.3f}")

# %% [markdown]
# Bei der direkten Zufallsstichprobe schwankt `z` schon bei gleichem `mu` stark: bei
# `mu = 2,00` zwischen 2,06 und 2,62, bei `mu = 2,01` zwischen 0,49 und 1,84. Diese Schwankung ist
# um ein Vielfaches größer als die Änderung von `mu` um 0,01. Einem einzelnen Wert von `z` lässt
# sich nicht ansehen, welcher Anteil auf `mu` zurückgeht; zwischen `mu` und genau diesem `z` gibt
# es keine feste Rechenvorschrift, und ohne Rechenvorschrift gibt es keine Ableitung. PyTorch gibt
# für diesen Weg deshalb den Gradienten 0 zurück. Bei `reparametrize` ist `eps` dagegen fest, und
# `z` ist eine feste Rechnung aus `mu`, `std` und `eps`: 0,01 mehr `mu` ergeben genau 0,01 mehr `z`
# (2,661 gegenüber 2,671), der Gradient ist also 1.
#
# Die Grafik zeigt denselben Unterschied für den gesamten Rückweg vom Verlust zu `mu` und
# `logvar`. Schwarze Pfeile zeigen die Vorwärtsrechnung, rote gestrichelte den Gradientenfluss:

# %%
from matplotlib.patches import FancyBboxPatch


def kasten(ax, x, y, text, farbe="white", breite=1.6):
    # Kasten mit Mittelpunkt (x, y); zurückgegeben werden linker und rechter Rand für die Pfeile
    ax.add_patch(FancyBboxPatch((x - breite / 2, y - 0.3), breite, 0.6, boxstyle="round,pad=0.04",
                                facecolor=farbe, edgecolor="black"))
    ax.text(x, y, text, ha="center", va="center", fontsize=10)
    return {"links": (x - breite / 2 - 0.05, y), "rechts": (x + breite / 2 + 0.05, y)}


def vorwaerts(ax, a, b):
    # Schwarzer Pfeil: Vorwärtsrechnung
    ax.annotate("", xy=b["links"], xytext=a["rechts"], arrowprops=dict(arrowstyle="->", lw=1.5))


def rueckwaerts(ax, a, b):
    # Roter gestrichelter Bogen: Gradientenfluss bei loss.backward()
    ax.annotate("", xy=a["rechts"], xytext=b["links"],
                arrowprops=dict(arrowstyle="->", lw=2, color="tab:red", linestyle="--",
                                connectionstyle="arc3,rad=-0.45"))


fig, (oben, unten) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [1, 1.5]})
for ax, hoehe in ((oben, 2.2), (unten, 3.4)):
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, hoehe)
    ax.axis("off")

# Weg 1: z direkt zufällig erzeugen
oben.set_title("Weg 1: direkte Zufallsstichprobe", loc="left", fontsize=12)
w1 = [kasten(oben, 1.2, 1.4, "mu, logvar", "lightblue"),
      kasten(oben, 4.6, 1.4, "Zufallsstichprobe aus der\nNormalverteilung (mu, std)", "khaki", breite=2.8),
      kasten(oben, 8.0, 1.4, "z", breite=0.9),
      kasten(oben, 10.4, 1.4, "Decoder"),
      kasten(oben, 13.0, 1.4, "Verlust")]
for a, b in zip(w1, w1[1:]):
    vorwaerts(oben, a, b)
for a, b in zip(w1, w1[1:]):
    rueckwaerts(oben, a, b)
# Der Gradientenfluss endet an der Zufallsstichprobe und erreicht mu und logvar nicht
oben.text(2.6, 1.12, "✗", color="tab:red", fontsize=26, ha="center", va="center",
          bbox=dict(facecolor="white", edgecolor="none", pad=0))
oben.text(2.6, 0.7, "kein Gradientenfluss", color="tab:red", ha="center", fontsize=10)

# Weg 2: reparametrize
unten.set_title("Weg 2: Reparametrisierung (reparametrize)", loc="left", fontsize=12)
eps_k = kasten(unten, 1.2, 2.8, "eps\n(standardnormalverteilt)", "khaki", breite=2.2)
logvar_k = kasten(unten, 1.2, 1.8, "logvar", "lightblue")
mu_k = kasten(unten, 1.2, 0.6, "mu", "lightblue")
std_k = kasten(unten, 3.8, 1.8, "std")
mal_k = kasten(unten, 6.3, 2.3, "Multiplikation", breite=1.6)
plus_k = kasten(unten, 8.3, 1.2, "Addition", breite=1.2)
z_k = kasten(unten, 9.9, 1.2, "z", breite=0.9)
dec_k = kasten(unten, 11.5, 1.2, "Decoder", breite=1.5)
loss_k = kasten(unten, 13.4, 1.2, "Verlust", breite=1.4)
for a, b in [(eps_k, mal_k), (logvar_k, std_k), (std_k, mal_k), (mal_k, plus_k), (mu_k, plus_k),
             (plus_k, z_k), (z_k, dec_k), (dec_k, loss_k)]:
    vorwaerts(unten, a, b)
for a, b in [(dec_k, loss_k), (z_k, dec_k), (plus_k, z_k), (mal_k, plus_k), (mu_k, plus_k),
             (std_k, mal_k), (logvar_k, std_k)]:
    rueckwaerts(unten, a, b)
unten.text(1.2, 2.33, "fester Zufallswert,\nkein trainierbarer Parameter", ha="center", va="center",
           fontsize=8.5, color="dimgray")

# Gemeinsame Legende
oben.plot([], [], color="black", label="Vorwärtsrechnung")
oben.plot([], [], color="tab:red", linestyle="--", label="Gradientenfluss bei loss.backward()")
oben.legend(loc="upper right", fontsize=9, frameon=False)
plt.tight_layout()
plt.show()

# %% [markdown]
# Bei der direkten Zufallsstichprobe (oberer Teil) bricht der Gradientenfluss an der Stichprobe ab.
# Für `mu` und `logvar` liefert die Rückwärtsrechnung keinen Gradienten; der Optimizer kann die
# Gewichte des Encoders, die diese beiden Größen erzeugen, daher nicht anpassen. Bei der
# Reparametrisierung (unterer Teil) ist die Zufallskomponente vollständig in `eps` ausgelagert.
# Da `eps` weder ein trainierbarer Parameter ist noch von `mu` oder `logvar` abhängt, besteht der
# Pfad vom Verlust zu `mu` und `logvar` ausschließlich aus differenzierbaren Operationen: Der
# Gradient gelangt über die Addition zu `mu` und über die Multiplikation und die Berechnung von
# `std` zu `logvar`. Die folgende Zelle überprüft dies, indem sie `z` auf beiden Wegen berechnet
# und die resultierenden Gradienten für `mu` und `logvar` ausgibt.

# %%
# Weg 1: z direkt zufällig erzeugen, mit Lage mu und Streuung std
mu = torch.tensor([0.0, 2.0, -1.0], requires_grad=True)
logvar = torch.tensor([0.0, -2.0, 1.0], requires_grad=True)
z = torch.normal(mu, logvar.mul(0.5).exp())
z.sum().backward()  # stellvertretend für einen Verlust, der von z abhängt
print("direkt erzeugt:  mu.grad =", mu.grad.tolist(), " logvar.grad =", logvar.grad.tolist())

# Weg 2: reparametrize
mu = torch.tensor([0.0, 2.0, -1.0], requires_grad=True)
logvar = torch.tensor([0.0, -2.0, 1.0], requires_grad=True)
torch.manual_seed(0)
std = logvar.mul(0.5).exp_()  # dieselben drei Zeilen wie oben
eps = torch.FloatTensor(std.size()).normal_().to(device)
z = eps.mul(std).add_(mu)
z.sum().backward()
print("reparametrize:   mu.grad =", mu.grad.tolist(),
      " logvar.grad =", [round(generator, 3) for generator in logvar.grad.tolist()])

device = device_before


# %% [markdown]
# Beim direkten Erzeugen kommen bei `mu` und `logvar` nur Nullen an. PyTorch rechnet zwar formal
# einen Gradienten aus, er ist aber immer 0; der Encoder würde nie erfahren, wie er `mu` und
# `logvar` verändern soll. Beim Weg über `reparametrize` ist `z` eine gewöhnliche Rechnung aus
# `mu`, `std` und einer festen Zahl `eps`:
#
# - Verschiebt man `mu` um 1, verschiebt sich `z` um 1; deshalb ist `mu.grad` überall 1.
# - Wie stark `z` auf `logvar` reagiert, hängt davon ab, wie weit `eps` gestreckt wird. Der
#   Gradient ist die Hälfte von `std` mal `eps`; im ersten Beispiel also die Hälfte von 1 mal
#   1,54, etwa 0,77.
#
# Das entspricht der Grafik oben: Der Zufall steckt ganz in `eps`, und `eps` hängt von keinem
# Parameter ab. Das Buch formuliert es so: Der Zufall liegt außerhalb des Netzes, der Gradient
# fließt um ihn herum.

# %% [markdown]
# ## 8. Der KL-Term in Zahlen, `reduction` und `beta`
#
# Abschnitt 7 hat gezeigt, wie der VAE aus `mu` und `logvar` einen zufälligen Punkt erzeugt.
# Damit allein wäre noch nichts gegen die Lücken aus Abschnitt 6 getan: Das Netz könnte die
# Glockenkurven beliebig schmal und weit verstreut machen und wäre dann wieder ein normaler
# Autoencoder. Deshalb gibt es den zweiten Teil der Verlustfunktion, den KL-Term. Er misst, wie
# stark die Verteilung des Encoders von der Standardnormalverteilung abweicht, also von einer
# Glockenkurve mit Mittelwert 0 und Streuung 1. Die Aufgabe gibt ihn als Klasse `KLDLoss` vor:

# %%
class KLDLoss(nn.Module):
    """Calculate Kullback-Leibler divergence between learned mean and logvar and standard normal distribution.
    This is *NOT* a re-implementation of torch.nn.KLDivLoss"""
    def __init__(self, reduction='sum'):
        super(KLDLoss, self).__init__()
        self.reduction = reduction

    def forward(self, mean, logvar):
        # KLD loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), 1)
        # Size average
        if self.reduction == 'mean':
            kld_loss = torch.mean(kld_loss)
        elif self.reduction == 'sum':
            kld_loss = torch.sum(kld_loss)
        return kld_loss


# %% [markdown]
# Statt die Rechnung in der Klasse auseinanderzunehmen, schauen wir auf ihre Werte. Wir geben ihr
# ein einzelnes "Bild" mit einer latenten Dimension und verändern `mu` und `logvar`:

# %%
kld_per_image = KLDLoss(reduction="none")  # weder summieren noch mitteln: ein Wert je Bild

print(f"{'mu':>5} {'logvar':>7} {'Streuung':>9} {'KL-Term':>8}")
for mean_value, log_var in [(0, 0), (1, 0), (2, 0), (3, 0), (0, -2), (0, -4), (0, 2), (2, -4)]:
    kl_value = kld_per_image(torch.tensor([[float(mean_value)]]), torch.tensor([[float(log_var)]])).item()
    std_value = torch.tensor(log_var / 2).exp().item()
    print(f"{mean_value:>5} {log_var:>7} {std_value:>9.2f} {kl_value + 0.0:>8.2f}")  # + 0.0 macht aus -0.00 eine 0.00


# %% [markdown]
# Die Tabelle lässt sich so lesen:
#
# - Bei `mu = 0` und `logvar = 0`, also genau der Standardnormalverteilung, ist der KL-Term 0.
# - Wandert der Mittelwert weg von 0, wächst der Term mit dem Quadrat des Abstands: 0,5 bei
#   Abstand 1, 2 bei Abstand 2, 4,5 bei Abstand 3. Weit verstreute Punkte wie in Abschnitt 6
#   werden also teuer.
# - Weicht die Streuung von 1 ab, wächst der Term ebenfalls, und zwar in beide Richtungen: bei
#   Streuung 0,37 auf etwa 0,6, bei 0,14 auf gut 1,5, bei 2,72 auf knapp 2,2. Eine sehr schmale
#   Glockenkurve wäre wieder fast ein fester Punkt, das verhindert der Term.
# - Kommt beides zusammen, addieren sich die Anteile: `mu = 2` mit Streuung 0,14 kostet 2 plus
#   gut 1,5.
# - Bei mehreren latenten Dimensionen addieren sich die Werte der einzelnen Dimensionen; das ist
#   die Summe über Achse 1 in `KLDLoss`.
#
# Zusammen mit dem Rekonstruktionsverlust entsteht ein Tauziehen: Die Rekonstruktion möchte für
# jedes Bild eine eigene, schmale, gut unterscheidbare Glockenkurve, der KL-Term möchte alle
# Glockenkurven zur Standardnormalverteilung zusammenschieben. `beta` legt fest, wie stark der
# KL-Term zieht.
#
# Bevor wir das in Abschnitt 9 ausprobieren, gibt es eine Falle, die mit `beta` nichts zu tun hat
# und doch dessen Wirkung bestimmt. Die Klasse `KLDLoss` summiert in der Grundeinstellung über
# alle Bilder des Batches (`reduction='sum'`). `nn.MSELoss` mittelt dagegen in der
# Grundeinstellung über alle Pixel aller Bilder (`reduction='mean'`). Wie weit das
# auseinanderliegt, zeigt ein Batch mit einem untrainierten VAE:

# %%
class SmallVAE(nn.Module):
    def __init__(self, latent_size, hidden_size=256):
        super().__init__()
        self.encoder = VaeEncoder(latent_size, hidden_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 784), nn.Sigmoid(), nn.Unflatten(1, (1, 28, 28)),
        )

    # reparametrize unverändert aus der Klasse VariationalAutoencoder der Aufgabe
    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_().to(device)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        x = self.reparametrize(mu, logvar)
        x = self.decoder(x)
        return x, mu, logvar


torch.manual_seed(0)
vae = SmallVAE(latent_size=2).to(device)
image, label = next(iter(train_loader))
image = image.to(device)
with torch.no_grad():
    output, mean, logvar = vae(image)
    rc_mean = nn.MSELoss()(output, image).item()  # Grundeinstellung: Mittelwert über alle Pixel
    rc_sum = nn.MSELoss(reduction="sum")(output, image).item()  # Summe über alle Pixel
    kl_sum = KLDLoss()(mean, logvar).item()  # Grundeinstellung: Summe über alle Bilder
    kl_mean = KLDLoss(reduction="mean")(mean, logvar).item()  # Mittelwert über die Bilder

print(f"MSELoss(), Mittel über {image.numel()} Pixel: {rc_mean:10.4f}")
print(f"MSELoss(reduction='sum'):              {rc_sum:10.1f}")
print(f"KLDLoss(), Summe über {image.size(0)} Bilder:     {kl_sum:10.2f}")
print(f"KLDLoss(reduction='mean'):             {kl_mean:10.4f}")


# %% [markdown]
# Ein Batch hat 64 × 784 = 50.176 Pixel. Daraus folgt:
#
# 1. Die Summe des Rekonstruktionsverlusts ist genau 50.176-mal so groß wie sein Mittelwert, die
#    Summe des KL-Terms 64-mal so groß wie sein Mittelwert. Beim untrainierten Netz ist der
#    KL-Term noch klein, weil `mu` und `logvar` alle nahe 0 liegen; er wächst erst, wenn der
#    Encoder lernt, die Bilder auseinanderzuziehen.
# 2. Kombiniert man `nn.MSELoss()` und `KLDLoss()` jeweils in der Grundeinstellung, wird nur der
#    Rekonstruktionsverlust durch 50.176 geteilt. Der KL-Term wiegt dann bei `beta = 1` so viel
#    wie bei `beta = 50.176`, wenn beide Teile summiert werden. Abschnitt 9 zeigt, dass bei zwei
#    Summen schon `beta = 20` zu viel ist.
# 3. Mit `reduction="sum"` bei beiden Teilen stehen Pixel und latente Dimensionen im selben
#    Verhältnis wie im VAE-Beispiel des Buchs, das beide Teile ebenfalls aufsummiert und einfach
#    addiert.
#
# Welcher Zahlenwert für `beta` sinnvoll ist, hängt also von der `reduction` der beiden Teile ab.
# Legen Sie diese zuerst fest und wählen Sie erst dann `beta`.
#
# Ein Nebeneffekt der Summen: Die Verlustwerte, die die Trainingsschleife der Aufgabe ausgibt,
# hängen dann von der Batch-Größe ab und liegen bei `batch_size = 512` in den Tausendern. Für den
# Vergleich von Trainings- und Testkurve spielt das keine Rolle.

# %% [markdown]
# ## 9. Die Wirkung von `beta`
#
# Mit beiden Verlustteilen aufsummiert, wie in Abschnitt 8 besprochen und wie im Buch, trainieren
# wir jetzt denselben kleinen VAE mit verschiedenen Werten für `beta`. Der latente Raum hat
# wieder 2 Dimensionen, damit wir ihn wie in den Abschnitten 5 und 6 zeichnen können. Die
# Trainingsschleife folgt der aus Teil 2 der Aufgabe:

# %%
def train_vae(beta, number_epochs=30, learning_rate=1e-3):
    torch.manual_seed(0)
    model = SmallVAE(latent_size=2).to(device)
    reconstruction_loss = nn.MSELoss(reduction="sum")
    kld_loss = KLDLoss(reduction="sum")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in tqdm(range(number_epochs), desc=f"beta = {beta}"):
        model.train()
        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device)
            output, mean, logvar = model(image)
            loss = reconstruction_loss(output, image) + beta * kld_loss(mean, logvar)
            loss.backward()
            optimizer.step()
    return model.eval()


# Hier ändern: welche Werte für beta verglichen werden
beta_values = [0, 1, 5, 20]
vaes = {beta: train_vae(beta) for beta in beta_values}

# %% [markdown]
# Für jedes `beta` messen wir zwei Dinge auf den Testbildern: wie gut die Rekonstruktion ist
# (mittlerer quadratischer Fehler je Pixel, decodiert aus `mu` ohne Zufall) und wie groß der
# KL-Term je Bild ist. Zum Vergleich steht der Autoencoder aus Abschnitt 2 mit 2 latenten
# Dimensionen dabei:

# %%
with torch.no_grad():
    ae_error = nn.MSELoss()(models[2](test_images.to(device)), test_images.to(device)).item()
print(f"{'':<12} {'Rekonstruktionsfehler':>22} {'KL-Term je Bild':>16}")
print(f"{'Autoencoder':<12} {ae_error:>22.4f} {'-':>16}")
for beta, vae in vaes.items():
    with torch.no_grad():
        mean, logvar = vae.encoder(test_images.to(device))
        error = nn.MSELoss()(vae.decoder(mean), test_images.to(device)).item()
        kl = KLDLoss(reduction="mean")(mean, logvar).item()
    print(f"{'beta = ' + str(beta):<12} {error:>22.4f} {kl:>16.2f}")

# %% [markdown]
# Bis `beta = 1` bleibt der Rekonstruktionsfehler etwa auf dem Niveau des Autoencoders, während
# der KL-Term von über 50 auf wenige Einheiten fällt. Danach kippt das Verhältnis: Bei `beta = 5`
# steigt der Fehler deutlich, und bei `beta = 20` ist der KL-Term praktisch 0. Was hinter diesen
# Zahlen steckt, zeigen der latente Raum und, wie in Abschnitt 6, ein Gitter aus decodierten
# Punkten. Das Gitter liegt jetzt für alle `beta` im selben festen Bereich von −2,5 bis 2,5 in
# beiden Koordinaten; dort landen fast alle Zufallswerte aus der Standardnormalverteilung.

# %%
fig, axes = plt.subplots(2, len(beta_values), figsize=(4.2 * len(beta_values), 8.5))
for column, (beta, vae) in enumerate(vaes.items()):
    with torch.no_grad():
        mean, _ = vae.encoder(test_images.to(device))
    mean = mean.cpu()
    for shape_id in range(3):
        mask = test_shapes == shape_id
        axes[0, column].scatter(mean[mask, 0], mean[mask, 1], s=3, color=colors[shape_id],
                                label=ALL_SHAPES[shape_id])
    # Gestrichelter Rahmen: der Bereich, den das Gitter darunter abdeckt
    axes[0, column].add_patch(plt.Rectangle((-2.5, -2.5), 5, 5, fill=False, linestyle="--", color="gray"))
    axes[0, column].set(title=f"beta = {beta}: mu der Testbilder", xlabel="Koordinate 1", ylabel="Koordinate 2")
    axes[1, column].imshow(decode_grid(vae.decoder, (-2.5, 2.5), (-2.5, 2.5)), cmap="gray", vmin=0, vmax=1)
    axes[1, column].set_title("Decoder-Ausgabe im Bereich ±2,5")
    axes[1, column].axis("off")
axes[0, 0].legend(markerscale=3, fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# In der oberen Zeile ist jeder Punkt der Mittelwert `mu` eines Testbildes, der gestrichelte
# Rahmen markiert den Bereich ±2,5. Die untere Zeile zeigt, was der Decoder an den Punkten eines
# 8 × 8-Gitters in diesem Rahmen erzeugt. Beachten Sie, dass die Achsen oben je nach `beta`
# verschieden skaliert sind. Von links nach rechts:
#
# - `beta = 0`: Ohne KL-Term verhält sich der VAE wie ein Autoencoder. Die Punkte streuen über
#   einen Bereich von mehreren Dutzend Einheiten, der Rahmen ±2,5 ist nur ein kleiner Ausschnitt
#   davon. Das Gitter zeigt deshalb fast nur eine einzige Form. Wer hier Punkte nach der
#   Standardnormalverteilung wählt, bekommt kaum Abwechslung; das ist das Problem aus Abschnitt 6
#   in anderer Gestalt.
# - `beta = 1`: Die Punkte liegen jetzt um den Ursprung und füllen den Rahmen weitgehend aus, die
#   drei Formen grenzen aneinander, statt in weit getrennten Ausläufern zu liegen. Im Gitter
#   erscheinen alle drei Formen in verschiedenen Größen, mit allmählichen Übergängen. So sieht
#   die "bessere Füllung des Raums" aus, die die Folien dem KL-Loss zuschreiben, bei kaum
#   schlechterer Rekonstruktion.
# - `beta = 5`: Der KL-Term drückt stärker. Kreise nehmen fast den ganzen Raum ein, Quadrate und
#   Balken werden in einen kleinen Bereich um den Ursprung gedrängt. Im Gitter gibt es fast nur
#   noch Kreise, und der Rekonstruktionsfehler steigt.
# - `beta = 20`: Alle Mittelwerte liegen als winziger Fleck auf dem Ursprung. Der Encoder gibt
#   für jedes Bild fast dieselbe Verteilung aus, der KL-Term ist 0, und der Decoder liefert
#   überall dasselbe verwaschene Durchschnittsbild. Dieser Zustand heißt Posterior Collapse: Der
#   latente Raum trägt keine Information mehr über das Eingabebild.
#
# Für Teil 2 der Aufgabe folgt daraus: Ein zu kleines `beta` lässt den latenten Raum so
# lückenhaft wie beim Autoencoder, ein zu großes macht die Rekonstruktionen unscharf, bis
# schließlich alle gleich aussehen. Wo der günstige Bereich liegt, hängt von den
# `reduction`-Einstellungen aus Abschnitt 8, von `latent_size` und von den Daten ab. Bei
# FashionMNIST mit mehr latenten Dimensionen können Sie den latenten Raum nicht direkt zeichnen.
# Vergleichen Sie dann die gespeicherten Rekonstruktionen für mehrere Werte von `beta`, und
# lassen Sie den Decoder zusätzlich Bilder aus zufälligen Punkten der Standardnormalverteilung
# erzeugen.

# %% [markdown]
# ## 10. Zusammenfassung
#
# - Ersatzbilder: Format (Batch, 1, 28, 28) mit Werten von 0 bis 1 wie FashionMNIST nach
#   `ToTensor()`. Der Autoencoder lernt nur aus den Pixeln; die Form dient nur zum Einfärben.
# - Engpass: Ohne Engpass lernt der Autoencoder, Pixel durchzureichen, und gibt auch nie gesehene
#   Kreuze erkennbar wieder. Mit kleinem latentem Raum muss er die Daten verdichten und zieht
#   unbekannte Formen zu bekannten hin. `latent_size` wägt Schärfe gegen Verdichtung ab.
# - Letzte Aktivierung und Verlust: ReLU ist nach oben offen, Tanh liefert −1 bis 1, Sigmoid 0
#   bis 1. Das Buch kombiniert Tanh mit auf −1 bis 1 normierten Bildern, die Aufgabe nutzt Bilder
#   von 0 bis 1. `to_image` der Aufgabe schneidet nur ab und verdeckt Werte außerhalb des
#   Bereichs. Softmax passt nicht, weil sich Grauwerte nicht zu 1 aufaddieren. `BCELoss` verlangt
#   Ausgaben zwischen 0 und 1, `MSELoss` nicht.
# - Formen: Conv2d ohne Padding mit Kernelgröße 5 macht aus 28 Pixeln 24, ConvTranspose2d mit
#   denselben Einstellungen macht daraus wieder 28. Bei Schrittweite 2 braucht der Rückweg
#   `output_padding=1`, um auf 28 statt 27 zu kommen. `Flatten` und `Unflatten` verbinden die
#   Faltungen mit der `Linear`-Schicht in den latenten Raum; deren Eingangsgröße ergibt sich aus
#   der Form nach der letzten Faltung.
# - Training: Trainings- und Testkurve fallen bei allen drei Netzen gleich ab, ein fehlender
#   Engpass ist nur an den Bildern zu erkennen. Mit BatchNorm gehören `model.train()` vor das
#   Training und `model.eval()` vor die Auswertung. Ein 2-dimensionaler latenter Raum trennt die
#   Formen, ohne dass das Netz sie kennt.
# - Zufällige Punkte: Der latente Raum eines Autoencoders hat Lücken, und seine Lage hängt vom
#   Training ab; in den Lücken erzeugt der Decoder Mischformen oder verzerrte Bilder.
# - VAE: Der Encoder liefert `mu` und `logvar`, den Logarithmus der Varianz. `reparametrize`
#   berechnet daraus die Streuung, erzeugt `eps` aus der Standardnormalverteilung und bildet `mu`
#   plus Streuung mal `eps`. Direktes zufälliges Erzeugen liefert nur Nullgradienten, über
#   `reparametrize` fließt der Gradient zu `mu` und `logvar`.
# - KL-Term: 0 für die Standardnormalverteilung, wächst mit dem Quadrat des Abstands von `mu` zu
#   0 und wenn die Streuung von 1 abweicht, egal in welche Richtung. `KLDLoss` summiert in der
#   Grundeinstellung über die Bilder, `MSELoss` mittelt über alle Pixel; die Wahl der `reduction`
#   bestimmt, welches `beta` sinnvoll ist.
# - `beta`: Bei 0 bleibt der latente Raum lückenhaft und weit verstreut, bei einem passenden Wert
#   füllt er den Bereich um den Ursprung mit allen Formen, bei zu großem Wert werden die
#   Rekonstruktionen unscharf, bis im Posterior Collapse alle Bilder gleich aussehen.
