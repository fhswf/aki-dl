# %% [markdown]
# # Ergänzung zu Termin 3: Rekurrente Netze für die Textklassifikation
#
# In der Aufgabe zu Termin 3 sollen Sie deutschsprachige Tweets aus GermEval 2018 als
# beleidigend oder nicht beleidigend einordnen. Dafür werden die Tweets bereinigt, mit spaCy in
# Wortvektoren übersetzt, zu Batches gepaddet und von einer GRU verarbeitet, deren letzter
# Zustand über eine lineare Schicht die Klasse liefert. Kapitel 5 erklärt die Bausteine dahinter:
# wie aus Text Vektoren werden, warum vortrainierte Embeddings bei wenig Daten helfen und wie
# RNN, LSTM und GRU eine Folge Schritt für Schritt verarbeiten. Wir folgen hier dem Weg eines
# Tweets durch die Aufgabe; rechts steht jeweils, in welchem Abschnitt und zu welcher
# Teilaufgabe der Schritt behandelt wird:
#
# ```
# "Wieso? Was findest du da unklar?"                    Text
#         │  Tokenisierung                              Abschnitt 1, Aufgabe 1.3
#         ▼
# ["Wieso", "?", "Was", ..., "?"]                       8 Tokens
#         │  Wortvektoren nachschlagen (spaCy)          Abschnitte 1 und 2, Aufgabe 1.4
#         ▼
# Matrix (8, 300)                                       ein Vektor je Token
#         │  mit anderen Tweets zum Batch auffüllen     Abschnitt 3, Aufgabe 1.5
#         ▼
# Tensor (64, längster Tweet, 300) + lengths
#         │  GRU liest Token für Token                  Abschnitte 4 bis 6, Aufgaben 1.6 und 1.7
#         ▼
# letzter Zustand h_n[-1], Form (64, hidden_dim)        ein Vektor je Tweet
#         │  lineare Schicht                            Aufgabe 1.6
#         ▼
# 2 Werte je Tweet (OTHER, OFFENSE)                     Training und Auswertung: Abschnitt 7, Aufgabe 1.8
# ```

# %% [markdown]
# ## 0. Setup

# %%
try:
    import torch  # torch ist auf Colab und dem Jupyter-Server meist schon installiert
    print(f"PyTorch {torch.__version__} bereits vorhanden, kein Neuinstall noetig.")
except ImportError:
    !pip install -q torch

# matplotlib für das Diagramm in Abschnitt 7, tqdm für den Fortschrittsbalken beim Training
!pip install -q matplotlib==3.11.2 tqdm==4.70.1

# %% [markdown]
# ## 1. Vom Tweet zur Vektorfolge
#
# Ein neuronales Netz rechnet mit Zahlen, nicht mit Zeichenketten. Bevor ein Tweet irgendein
# Netz erreicht, muss er deshalb in Einheiten zerlegt werden (Tokenisierung) und jede Einheit
# muss einen Vektor bekommen (Vektorisierung). Wie fein zerlegt wird, ist dabei keine
# Nebensache. Ein Tweet aus den Trainingsdaten der Aufgabe zeigt das:

# %%
import re

tweet = "Wieso? Was findest du da unklar?"  # Tweet Nr. 7 aus germeval2018.training.tsv

# Variante 1: nur an Leerzeichen trennen
print(tweet.split())


# Variante 2: Wörter und Satzzeichen getrennt, so wie spaCy es macht.
# Der reguläre Ausdruck hat zwei Alternativen, getrennt durch |:
#   \w+      eine zusammenhängende Folge von Wortzeichen (Buchstaben, Ziffern, Unterstrich)
#   [^\w\s]  genau ein Zeichen, das weder Wortzeichen noch Leerraum ist, also ein Satzzeichen
# re.findall liefert alle Treffer in der Reihenfolge, in der sie im Text stehen.
def tokenisiere(text):
    return re.findall(r"\w+|[^\w\s]", text)


print(tokenisiere(tweet))

# %% [markdown]
# `split()` trennt nur an Leerzeichen und liefert 6 Tokens, das Fragezeichen klebt am Wort.
# Der Tokenizer trennt Satzzeichen ab und kommt auf 8 Tokens. spaCy verhält sich in der Aufgabe
# genauso, deshalb erzeugt `vectorize()` für diesen Tweet 8 statt 6 Wortvektoren. Das hat einen
# praktischen Vorteil: `unklar?` und `unklar` wären sonst zwei verschiedene Wörter mit zwei
# verschiedenen Vektoren. Aus demselben Grund ersetzt `clean_tweet()` aus Aufgabe 1.3 Bindestriche durch
# Leerzeichen, damit Komposita mit Bindestrich in bekannte Einzelwörter zerfallen, statt als
# unbekanntes Token ganz ohne Vektor herauszufallen.
#
# Für die Vektorisierung selbst gibt es zwei Wege. Beim One-Hot-Encoding bekommt jedes Wort des
# Vokabulars einen Vektor der Länge N (Größe des Vokabulars) mit genau einer 1:

# %%
import torch

# Vokabular: jedes verschiedene Token genau einmal ("?" kommt zweimal vor, zählt aber nur einfach)
vokabular = sorted(set(tokenisiere(tweet)))
# Jedes Wort bekommt eine feste Nummer, seine Position im Vokabular
woerter_index = {wort: i for i, wort in enumerate(vokabular)}
# Einheitsmatrix N × N: Zeile i hat genau an Stelle i eine 1, das ist der One-Hot-Vektor von Wort i
one_hot = torch.eye(len(vokabular))

print(vokabular)
print(one_hot[woerter_index["unklar"]])  # eine 1 an der Position von "unklar", sonst Nullen
# Skalarprodukt zweier verschiedener One-Hot-Vektoren: die Einsen liegen nie an derselben Stelle
print(one_hot[woerter_index["Was"]] @ one_hot[woerter_index["Wieso"]])

# %% [markdown]
# Bei 7 verschiedenen Tokens ist das noch harmlos. Zwei Eigenschaften machen One-Hot aber für
# echte Texte unbrauchbar. Erstens wächst die Vektorlänge mit dem Vokabular: Bei 20.000
# Wörtern braucht man 20.000 × 20.000 Zahlen, fast alle davon 0. Zweitens ist das Skalarprodukt
# zweier verschiedener Wörter immer 0, wie die letzte Zeile zeigt. "Wieso" ist "Was" also
# genauso unähnlich wie jedem beliebigen anderen Wort; Ähnlichkeit zwischen Wörtern kann diese
# Darstellung gar nicht ausdrücken.
#
# Word Embeddings lösen beide Probleme mit dichten Vektoren fester Länge, bei spaCys
# `de_core_news_md` sind es 300 Dimensionen. Woher kommen diese 300 Zahlen pro Wort? Sie
# wurden vorab auf sehr großen deutschen Textsammlungen gelernt. Zu Beginn hat jedes Wort
# einen zufälligen Vektor; dann wird ein Netz auf eine Hilfsaufgabe trainiert, etwa ein Wort
# aus seinen Nachbarwörtern vorherzusagen, und die Vektoren werden dabei mit angepasst. Wörter,
# die in ähnlichen Zusammenhängen stehen ("Das Essen war toll / super / prima"), müssen für
# diese Aufgabe ähnliche Vektoren haben und rücken deshalb im Lauf des Trainings zusammen.
# Die einzelnen Zahlen eines Vektors haben dabei keine ablesbare Bedeutung, aussagekräftig ist
# nur, wie die Vektoren zueinander liegen. spaCy liefert das Ergebnis dieses Trainings als
# fertige Nachschlagetabelle: Wort hinein, 300 Zahlen heraus.
#
# Das spaCy-Modell müsste dafür erst heruntergeladen werden. Stattdessen schreiben wir hier eine
# eigene kleine Funktion `wortvektor(wort)`, die wie spaCy zu jedem Wort 300 Zahlen liefert.
# Die Ähnlichkeit zwischen Wörtern stellen wir dabei von Hand her, statt sie zu lernen. Dafür
# legen wir zwei Bedeutungsgruppen fest, Lob ("toll", "super", "prima") und Beleidigung
# ("Trottel", "Dummkopf", "Idiot"), und bauen die Vektoren in zwei Schritten:
#
# 1. Jede Gruppe bekommt einen eigenen Zufallsvektor, ihren Grundvektor. Es gibt also einen
#    Lob-Grundvektor und einen Beleidigungs-Grundvektor, beide unabhängig voneinander gezogen.
# 2. Jedes Wort einer Gruppe ist der Grundvektor seiner Gruppe plus ein kleinerer Zufallsanteil,
#    der nur zu diesem Wort gehört.
#
# Damit zeigen "toll", "super" und "prima" alle ungefähr in die Richtung des Lob-Grundvektors
# und liegen deshalb nah beieinander; für die Schimpfwörter gilt dasselbe mit dem
# Beleidigungs-Grundvektor. Dass die beiden Gruppen weit auseinander liegen, legen wir dagegen
# nirgends ausdrücklich fest. Das ergibt sich von selbst, weil die beiden Grundvektoren
# unabhängig gezogene Zufallsvektoren mit 300 Einträgen sind; warum das genügt, zeigen wir
# gleich nach dem Code. Wörter ohne Gruppe wie "Tisch" bekommen einen reinen Zufallsvektor und
# liegen aus demselben Grund weit weg von allen anderen.
#
# Die Gruppen legen wir selbst fest; bei spaCy ergeben sie sich aus den Trainingstexten.

# %%
import zlib

EMBEDDING_DIM = 300  # gleiche Dimension wie die spaCy-Vektoren in der Aufgabe

# Unsere von Hand festgelegten Bedeutungsgruppen (bei spaCy entstehen sie durch das Training)
bedeutungsgruppen = {
    "lob": ["toll", "super", "prima"],
    "beleidigung": ["Trottel", "Dummkopf", "Idiot"],
}
# Umgekehrtes Nachschlagen: zu jedem Wort seine Gruppe, z. B. gruppe_von["super"] == "lob"
gruppe_von = {wort: gruppe for gruppe, woerter in bedeutungsgruppen.items() for wort in woerter}


def _zufallsvektor(schluessel):
    # Seed aus dem Text selbst: derselbe Schlüssel ergibt bei jedem Aufruf denselben Vektor
    seed = zlib.crc32(schluessel.encode())
    # Eigener Generator, damit der globale Zufallszustand (torch.manual_seed) unberührt bleibt
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(EMBEDDING_DIM, generator=generator)  # 300 Werte, standardnormalverteilt


def wortvektor(wort):
    if wort in gruppe_von:
        grundvektor = _zufallsvektor("gruppe:" + gruppe_von[wort])  # für alle Wörter der Gruppe gleich
        eigenanteil = _zufallsvektor(wort)  # nur für dieses Wort
        return grundvektor + 0.5 * eigenanteil  # gemeinsame Richtung + kleine individuelle Abweichung
    return _zufallsvektor(wort)  # Wort ohne Gruppe: eigene, zufällige Richtung


print(wortvektor("Trottel").shape)  # 300 Zahlen pro Wort
print(wortvektor("Trottel")[:5])  # die ersten 5 davon; einzeln haben sie keine Bedeutung
print(torch.equal(wortvektor("Trottel"), wortvektor("Trottel")))  # zweimal gleich übersetzt?

# %% [markdown]
# Jedes Wort wird zu einem Vektor mit 300 Einträgen, und ein zweiter Aufruf liefert exakt
# denselben Vektor. Das ist die Mindestanforderung an ein Embedding: Ein Wort muss immer gleich
# übersetzt werden, sonst könnte das Netz nichts über es lernen.
#
# Für die folgenden Abschnitte reicht es, `wortvektor()` als Nachschlagetabelle zu sehen, in der
# Wörter einer Gruppe ähnliche Einträge haben. Die Umsetzung im Detail:
#
# - Der Seed wird mit `zlib.crc32` aus dem Wort berechnet. Pythons eingebautes `hash()` wäre
#   hier ungeeignet, weil es für Zeichenketten bei jedem Start des Interpreters anders ausfällt;
#   nach einem Kernel-Neustart hätte jedes Wort dann einen anderen Vektor.
# - `torch.Generator()` ist ein eigener Zufallsgenerator nur für diesen Aufruf. Ein
#   `torch.manual_seed` an dieser Stelle würde dagegen den globalen Zufallszustand
#   zurücksetzen und damit beeinflussen, wie später Gewichte initialisiert und Batches gemischt
#   werden.
# - Der Faktor 0,5 steuert, wie ähnlich sich Wörter einer Gruppe sind. Je kleiner der
#   wortspezifische Anteil im Verhältnis zum gemeinsamen Grundvektor, desto ähnlicher die
#   Wörter. Mit 0,5 landet die Ähnlichkeit innerhalb einer Gruppe bei etwa 0,8, mit 1 wären es
#   nur noch etwa 0,5, mit 0 wären alle Wörter einer Gruppe identisch.
#
# Offen ist noch, warum zwei unabhängige Zufallsvektoren, etwa der Lob- und der
# Beleidigungs-Grundvektor, weit auseinander liegen. Als Maß für "nah" und "weit" verwenden wir
# die Kosinus-Ähnlichkeit. Sie misst, wie sehr zwei Vektoren in dieselbe Richtung zeigen, und
# ignoriert dabei ihre Länge. 1 heißt gleiche Richtung, 0 heißt senkrecht zueinander, also ohne
# erkennbaren Zusammenhang, −1 heißt entgegengesetzt. Wir ziehen je 1.000 Paare
# unabhängiger Zufallsvektoren mit 2, 10 und 300 Einträgen und schauen, wie groß ihre
# Kosinus-Ähnlichkeit typischerweise ist:

# %%
import torch.nn.functional as F

generator = torch.Generator().manual_seed(0)
for dim in [2, 10, 300]:
    a = torch.randn(1000, dim, generator=generator)  # 1.000 Zufallsvektoren mit dim Einträgen
    b = torch.randn(1000, dim, generator=generator)  # 1.000 weitere, unabhängig von a
    cos = F.cosine_similarity(a, b, dim=1)  # Kosinus-Ähnlichkeit je Paar (a[i], b[i])
    print(f"{dim:>3} Dimensionen: mittlerer Betrag {cos.abs().mean():.2f}")

# Und konkret für unsere beiden Grundvektoren:
lob_grund = _zufallsvektor("gruppe:lob")
beleidigung_grund = _zufallsvektor("gruppe:beleidigung")
print(f"Lob- gegen Beleidigungs-Grundvektor: {F.cosine_similarity(lob_grund, beleidigung_grund, dim=0):.2f}")

# %% [markdown]
# In 2 Dimensionen haben zwei zufällige Vektoren oft einen deutlichen Zusammenhang, in 300
# Dimensionen praktisch nie: Der mittlere Betrag der Kosinus-Ähnlichkeit fällt auf etwa 0,05.
# Der Grund liegt darin, wie die Kosinus-Ähnlichkeit berechnet wird: Die beiden Vektoren werden
# Eintrag für Eintrag multipliziert und die 300 Produkte aufsummiert. Bei zwei unabhängigen
# Zufallsvektoren ist jedes dieser Produkte mal positiv, mal negativ, und in der Summe heben sie
# sich größtenteils auf. Je mehr Einträge, desto gründlicher gleicht sich das aus. Zufällige
# Richtungen in einem
# hochdimensionalen Raum stehen also fast immer nahezu senkrecht aufeinander, und genau das
# trennt unsere beiden Gruppen, ohne dass wir dafür etwas tun müssen.
#
# Innerhalb einer Gruppe ist es anders: Dort teilen die Wörter den Grundvektor, und dieser
# gemeinsame Anteil sorgt in jedem Eintrag für Produkte mit gleichem Vorzeichen, die sich nicht
# aufheben, sondern aufaddieren. Beides zusammen
# prüfen wir jetzt für alle Paare aus fünf Beispielwörtern auf einmal:

# %%
# Je zwei Wörter aus beiden Gruppen und "Tisch" als Wort ohne Gruppe
beispielwoerter = ["toll", "super", "Trottel", "Dummkopf", "Tisch"]
vektoren = torch.stack([wortvektor(w) for w in beispielwoerter])  # 5 Vektoren übereinander: (5, 300)

# (5, 1, 300) gegen (1, 5, 300): Broadcasting bildet alle 5 × 5 Paare,
# dim=2 vergleicht entlang der 300 Einträge, Ergebnis ist eine 5 × 5-Matrix
aehnlichkeit = F.cosine_similarity(vektoren.unsqueeze(1), vektoren.unsqueeze(0), dim=2)

# Ausgabe als Tabelle: jede Spalte 9 Zeichen breit, rechtsbündig
print(" " * 9 + "".join(f"{w:>9}" for w in beispielwoerter))  # Kopfzeile
for wort, zeile in zip(beispielwoerter, aehnlichkeit):
    print(f"{wort:>9}" + "".join(f"{wert:9.2f}" for wert in zeile))

# %% [markdown]
# Zeile und Spalte geben jeweils das Wortpaar an, die Diagonale ist trivialerweise 1. Innerhalb
# einer Gruppe liegt die Ähnlichkeit wie berechnet bei etwa 0,8, weil die Wörter ihren
# Grundvektor teilen. Zwischen den Gruppen und zum Wort "Tisch" liegt sie nahe 0, weil hier nur
# unabhängige Zufallsvektoren aufeinandertreffen. Genau diese Struktur bringen die echten spaCy-Vektoren mit, nur eben aus
# großen Textmengen gelernt statt von Hand gesetzt. Allerdings sind die echten Gruppen nicht so
# sauber getrennt wie unsere: Gegensätze wie "gut" und "schlecht" stehen in Texten oft an
# denselben Stellen ("Das Essen war gut / schlecht") und liegen deshalb in gelernten Embeddings
# häufig sogar recht nah beieinander. Für die Klassifikation ist das wertvoll:
# Hat das Netz gelernt, dass "Trottel" auf eine Beleidigung hinweist, liegt ein selten
# gesehenes Wort wie "Dummkopf" im Vektorraum gleich daneben und wird ähnlich behandelt. Bei
# 20.000 Wörtern und 300 Dimensionen sind das außerdem nur 20.000 × 300 statt 20.000 × 20.000
# Zahlen.

# %% [markdown]
# ## 2. Vortrainierte Embeddings: im Modell oder davor?
#
# In Abschnitt 1 haben wir mit `wortvektor(wort)` eine Nachschlagetabelle gebaut, die jedem
# Wort 300 Zahlen zuordnet. In diesem Notebook übernimmt sie die Rolle, die in der Aufgabe
# spaCys `tok.vector` spielt. Gemeinsam ist beiden: Die Vektoren stehen fest, bevor unser
# Klassifikator überhaupt existiert, und der Klassifikator soll sie nur benutzen, nicht
# verändern. Bei spaCy, weil die Vektoren schon auf großen Textmengen gelernt wurden
# ("vortrainiert"), bei uns, weil wir sie von Hand festgelegt haben. Offen ist noch, an welcher
# Stelle der Pipeline Wörter in diese Vektoren übersetzt werden: im Modell oder davor.
#
# Die übliche Lösung, wenn man Embeddings selbst trainiert, ist eine `nn.Embedding`-Schicht im
# Modell. Sie ist eine Tabelle (ein Tensor) mit einer Zeile pro Wort des Vokabulars. Ein Tensor
# kann aber keine Zeichenketten enthalten, also bekommt die Schicht statt eines Wortes die
# Nummer seiner Zeile. Diese Nummern haben wir schon: `woerter_index` aus Abschnitt 1 ordnet
# jedem Wort seine Position im Vokabular zu. Für vortrainierte Vektoren werden die Zeilen der
# Tabelle mit den fertigen Vektoren belegt und eingefroren, damit das Training sie nicht verändert.

# %%
import torch.nn as nn

# Nachschlagetabelle: Zeile i enthält den Vektor von Wort i aus dem Vokabular, Form (7, 300)
vektormatrix = torch.stack([wortvektor(wort) for wort in vokabular])
# Embedding-Schicht mit genau diesen Zeilen als Gewichten; freeze=True setzt requires_grad=False
embedding = nn.Embedding.from_pretrained(vektormatrix, freeze=True)

# Eine nn.Embedding-Schicht erwartet Wortnummern, keine Zeichenketten: Tweet -> Tokens -> Indizes
indizes = torch.tensor([woerter_index[token] for token in tokenisiere(tweet)])
print(indizes)
print(embedding(indizes).shape)  # je Token eine Zeile der Tabelle: (8, 300)
print(embedding.weight.requires_grad)  # False: das Training darf die Tabelle nicht verändern

# %% [markdown]
# Die erste Zeile zeigt die Zeilennummern für die 8 Tokens: "Wieso" steht im Vokabular an
# Position 2, "?" an Position 0 (deshalb taucht die 0 zweimal auf). Für jede Nummer liefert die
# Schicht die passende Zeile, zusammen eine (8, 300)-Matrix. `freeze=True` sorgt dafür, dass das
# Training diese Zeilen nicht verändert.
#
# Die Aufgabe geht einen anderen Weg und kommt ganz ohne `nn.Embedding` aus. Dort steckt das
# Nachschlagen der Vektoren in `vectorize()`, und das läuft in `collate_batch()`, also noch vor
# dem Modell. Hier dieselbe Funktion, nur mit `wortvektor()` statt spaCy:

# %%
# Entspricht vectorize() aus der Aufgabe: dort tok.vector von spaCy, hier unser wortvektor()
def vectorize(text):
    # Tokens nachschlagen und die Vektoren zu einer Matrix (Anzahl Tokens, 300) stapeln
    return torch.stack([wortvektor(token) for token in tokenisiere(text)])


vektorfolge = vectorize(tweet)
print(vektorfolge.shape, vektorfolge.requires_grad)
# Gleiches Ergebnis wie das Nachschlagen über die Embedding-Schicht?
print(torch.equal(vektorfolge, embedding(indizes)))

# %% [markdown]
# Beide Wege liefern exakt dieselben Vektoren. Der Unterschied liegt darin, was das Modell davon
# weiß: Bei `vectorize()` kommen die Vektoren als gewöhnliche Eingabedaten im Modell an, ohne
# `requires_grad`. Sie sind damit automatisch eingefroren, ohne dass man etwas dafür tun muss,
# und sie tauchen auch nicht unter `model.parameters()` auf. Für das Zählen der trainierbaren
# Parameter in der Aufgabe heißt das: Gezählt werden nur GRU und lineare Schicht, die
# 300-dimensionalen spaCy-Vektoren gehören nicht dazu. Mit einer eingefrorenen
# `nn.Embedding`-Schicht wäre es ebenso, nur dann wegen `requires_grad=False`:

# %%
# Minimalmodell: eingefrorene Embedding-Schicht, dahinter eine lineare Schicht auf 2 Klassen
mit_embedding = nn.Sequential(embedding, nn.Linear(EMBEDDING_DIM, 2))

# numel() = Anzahl Einträge eines Tensors; einmal alle Parameter, einmal nur die trainierbaren
alle = sum(p.numel() for p in mit_embedding.parameters())
trainierbar = sum(p.numel() for p in mit_embedding.parameters() if p.requires_grad)
print(alle, trainierbar)

# %% [markdown]
# Die 2.702 Parameter setzen sich so zusammen: Die Embedding-Tabelle hat 7 Zeilen (eine je Wort
# im Vokabular) mit je 300 Werten, also 2.100. Die lineare Schicht hat 300 × 2 Gewichte plus 2
# Bias-Werte, also 602. Trainierbar sind nur diese 602; die 2.100 Werte der Tabelle bleiben
# eingefroren und werden vom Optimizer nie verändert.
#
# Warum einfrieren und nicht einfach mittrainieren? Bei unserem Mini-Vokabular spielt das keine
# Rolle, bei GermEval sehr wohl. Zerlegt man die rund 5.000 Trainings-Tweets mit dem Tokenizer
# aus Abschnitt 1, kommen etwa 17.500 verschiedene Tokens zusammen. Eine trainierbare
# Embedding-Tabelle dafür hätte 17.500 × 300, also gut 5 Millionen Parameter, mehr als
# tausendmal so viele wie es Tweets gibt. Dazu kommt, wie ungleich die Wörter verteilt sind:
# Rund 11.000 dieser Tokens kommen im gesamten Trainingsset genau einmal vor. Aus einem einzigen
# Tweet lässt sich nicht lernen, wo ein Wort in einem 300-dimensionalen Raum hingehört; das Netz
# würde sich den Vektor so zurechtlegen, dass genau dieser eine Tweet richtig klassifiziert wird,
# also auswendig lernen. Und gut die Hälfte der verschiedenen Tokens im Testset taucht im
# Training überhaupt nicht auf. Ihre Vektoren würden nie angepasst und blieben auf ihren
# zufälligen Startwerten stehen.
#
# Vortrainierte Vektoren umgehen beide Probleme. spaCy hat sie auf einer Textmenge gelernt, die
# um Größenordnungen größer ist als GermEval, dort kamen auch seltene Wörter oft genug vor. Ein
# Wort, das in den Trainings-Tweets nie auftaucht, hat trotzdem einen sinnvollen Vektor in der
# Nähe bedeutungsverwandter Wörter; genau das werden wir in Abschnitt 7 mit "Idiot" sehen. Das
# Einfrieren schützt dieses Wissen: Würde man die Vektoren mittrainieren, könnten die wenigen
# GermEval-Tweets sie wieder verbiegen. Das Prinzip kennen Sie aus Termin 2, wo ein auf ImageNet
# vortrainiertes CNN seine Merkmale mitbrachte und nur der neue Kopf trainiert wurde.
#
# Mittrainieren lohnt sich erst, wenn viele Daten vorhanden sind oder das Vokabular stark von
# allgemeiner Sprache abweicht, etwa bei medizinischen Fachtexten. Dann startet man meist mit den
# vortrainierten Vektoren und lässt sie mit kleiner Lernrate nachjustieren, statt bei null
# anzufangen.

# %% [markdown]
# ## 3. Batches aus unterschiedlich langen Texten
#
# Nach Abschnitt 2 ist jeder Tweet eine Matrix der Form (Anzahl Tokens, 300). Die Anzahl Tokens
# ist aber von Tweet zu Tweet verschieden, und ein Batch muss ein einziger Tensor sein. Für die
# folgenden Abschnitte brauchen wir dazu einen kleinen Datensatz. Statt der GermEval-Tweets
# verwenden wir kurze, künstlich erzeugte Sätze im selben Format: `Record` mit Text, Haupt- und
# Nebenlabel, die Hauptlabels `OFFENSE` und `OTHER`. Ein Satz gilt als beleidigend, wenn er ein
# Schimpfwort enthält, das nicht direkt durch "kein" verneint wird. "du bist ein Trottel" ist
# also `OFFENSE`, "du bist kein Trottel" dagegen `OTHER`. Weil "kein" auch in harmlosen
# Wendungen wie "kein Problem" vorkommt, reicht es nicht, nur auf einzelne Wörter zu achten;
# es kommt auf die Reihenfolge an. Mit rund einem Drittel beleidigender Sätze liegt die
# Klassenverteilung ähnlich wie bei GermEval.
#
# Zwei Details machen die künstlichen Sätze realistischer. Erstens ist bei 15 % der
# Trainingssätze das Label vertauscht; auch menschliche Annotationen sind nie ganz einheitlich,
# und ein Modell soll die Regel lernen, nicht die Fehler. Die Testsätze bleiben fehlerfrei,
# damit sie messen, was das Modell tatsächlich gelernt hat. Zweitens kommt das Schimpfwort
# "Idiot" nur in den Testsätzen vor. Ob das Modell es trotzdem richtig einordnet, hängt allein
# davon ab, dass sein Vektor nahe bei "Trottel" und "Dummkopf" liegt (Abschnitt 1).

# %%
import random
from collections import Counter, namedtuple

# Gleiches Format wie in der Aufgabe: Text, Hauptlabel (OFFENSE/OTHER), Nebenlabel
Record = namedtuple("Record", ["text", "primary_label", "secondary_label"])

# Satzbausteine: Anfang + "ein"/"kein" + Nomen + Ende
anfaenge = ["du bist", "der Typ ist", "ehrlich , das ist", "mein Nachbar ist", "sie ist wirklich"]
neutrale_nomen = ["Lehrer", "Mensch", "Fan", "Profi", "Nachbar"]
enden = ["", "!", "?", "und das weiß jeder", "heute wieder"]


def erzeuge_satz(rng, schimpfwoerter, rauschen):
    beleidigend_gemeint = rng.random() < 0.5  # halbe-halbe: Schimpfwort oder neutrales Nomen
    nomen = rng.choice(schimpfwoerter if beleidigend_gemeint else neutrale_nomen)
    artikel = "ein" if rng.random() < 0.67 else "kein"  # in einem Drittel der Fälle verneint
    teile = [rng.choice(anfaenge), artikel, nomen, rng.choice(enden)]
    if rng.random() < 0.3:
        teile.insert(0, "kein Problem ,")  # "kein" ohne Bezug zum Schimpfwort
    text = " ".join(t for t in teile if t)  # leeres Ende "" überspringen
    # Die Regel: beleidigend genau dann, wenn Schimpfwort UND nicht direkt verneint
    offensiv = beleidigend_gemeint and artikel == "ein"
    if rng.random() < rauschen:
        offensiv = not offensiv  # Annotationsfehler simulieren
    if offensiv:
        return Record(text, "OFFENSE", "INSULT")
    return Record(text, "OTHER", "OTHER")


rng = random.Random(0)  # fester Seed: bei jedem Lauf dieselben Sätze
# Training: nur zwei Schimpfwörter, 15 % vertauschte Labels; Test: zusätzlich "Idiot", fehlerfrei
training_data = [erzeuge_satz(rng, ["Trottel", "Dummkopf"], rauschen=0.15) for _ in range(300)]
test_data = [erzeuge_satz(rng, ["Trottel", "Dummkopf", "Idiot"], rauschen=0.0) for _ in range(300)]

print("Die ersten vier Trainingssätze:")
for record in training_data[:4]:
    print(" ", record)

# Unter den ersten vier ist kein beleidigender Satz, daher gezielt die ersten zwei mit OFFENSE
print("Die ersten zwei Trainingssätze mit OFFENSE:")
for record in [r for r in training_data if r.primary_label == "OFFENSE"][:2]:
    print(" ", record)

print(Counter(r.primary_label for r in training_data))  # Klassenverteilung im Training

# %% [markdown]
# Die beiden `OFFENSE`-Sätze zeigen den Normalfall: ein Schimpfwort mit "ein". Der dritte der
# ersten vier Sätze enthält dasselbe Schimpfwort, aber mit "kein", und ist deshalb `OTHER`. Der
# zweite ist einer der absichtlich vertauschten Fälle: "ein Dummkopf" ohne Verneinung, aber als
# `OTHER` markiert.
#
# Die Sätze sind zwischen 4 und 13 Tokens lang. `collate_batch` ist unverändert aus der Aufgabe
# übernommen, nur `vectorize` ist unsere Version aus Abschnitt 2, die `wortvektor()` statt
# spaCy verwendet:

# %%
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

LABEL = {"OFFENSE": 1, "OTHER": 0}  # Klassennamen -> Zahlen für CrossEntropyLoss


# Wird vom DataLoader mit einer Liste von Records aufgerufen und baut daraus einen Batch
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for record in batch:
        label_list.append(LABEL[record.primary_label])
        processed_text = vectorize(record.text)  # (Anzahl Tokens, 300), je Tweet verschieden lang
        text_list.append(processed_text)
        lengths.append(processed_text.shape[0])  # echte Länge merken, bevor aufgefüllt wird
    # pad_sequence füllt alle Matrizen mit Nullzeilen auf die Länge der längsten auf
    return torch.tensor(label_list), pad_sequence(text_list, batch_first=True), lengths


# generator mit festem Seed nur, damit die gezeigten Batches bei jedem Lauf gleich sind
demo_loader = DataLoader(training_data, batch_size=4, shuffle=True, collate_fn=collate_batch,
                         generator=torch.Generator().manual_seed(0))

for labels, texts, lengths in list(demo_loader)[:3]:  # die ersten drei Batches ansehen
    print(labels.shape, texts.shape, lengths)

# %% [markdown]
# Die Batch-Größe ist immer 4, die Vektordimension immer 300, nur die mittlere Dimension
# schwankt. `pad_sequence` füllt jeden Tweet mit Nullvektoren auf, bis er so lang ist wie der
# längste Tweet *in diesem Batch*. Die Form eines Batches hängt also davon ab, welche Tweets
# `shuffle=True` gerade zusammenwürfelt. Dass die aufgefüllten Positionen tatsächlich nur
# Nullen enthalten, lässt sich direkt prüfen:

# %%
# Ein neuer Durchlauf durch demo_loader mischt neu, daher ist das ein anderer Batch als oben
labels, texts, lengths = next(iter(demo_loader))
# Je Position: Summe der Beträge über die 300 Einträge. Genau 0 heißt: reiner Füllvektor.
# Danach je Tweet zählen, wie viele Positionen das sind.
nullzeilen = (texts.abs().sum(dim=2) == 0).sum(dim=1)

print(lengths)  # echte Längen
print(nullzeilen.tolist())  # gezählte Füllpositionen
print([texts.size(1) - laenge for laenge in lengths])  # erwartet: Batch-Länge minus echte Länge

# %% [markdown]
# Die Anzahl der Nullvektoren je Tweet ist genau die Differenz zwischen Batch-Länge und
# eigener Länge. Das Buch geht einen anderen Weg: Dort wird jeder Text beim Einlesen mit
# `fix_length` auf eine feste Länge (20 bzw. 200 Tokens) abgeschnitten oder aufgefüllt. Das
# ergibt immer gleich geformte Batches, kostet aber doppelt: Lange Texte verlieren ihr Ende,
# kurze Texte schleppen viele Füllpositionen mit. `pad_sequence` füllt nur so weit auf, wie der
# jeweilige Batch es erfordert, und `lengths` merkt sich, wo in jeder Zeile der echte Text
# endet. Diese Liste wird in Abschnitt 5 entscheidend.

# %% [markdown]
# ## 4. Was im rekurrenten Netz passiert
#
# Mit den gepaddeten Batches aus Abschnitt 3 liegen die Eingaben in der Form vor, die ein
# rekurrentes Netz erwartet: (Batch, Position, Merkmale). Die mittlere Achse ist die Position
# des Tokens im Tweet. In der Literatur und in der PyTorch-Dokumentation heißt ihre Länge
# Sequenzlänge und ein einzelner Schritt darauf Zeitschritt, weil man sich die Folge als Ablauf
# vorstellt, in dem ein Element nach dem anderen eintrifft. Mit echter Zeit hat das bei Text
# nichts zu tun: Der fünfte Zeitschritt ist einfach das fünfte Wort.
#
# Was das Netz entlang dieser Achse macht, lässt sich am besten an einer einzelnen Zelle von
# Hand nachvollziehen. Ein RNN liest die Folge Wort für Wort und trägt dabei einen
# Zustandsvektor, den Hidden State, mit sich. In jedem Schritt passiert dasselbe:
#
# 1. Der Wortvektor wird mit einer Gewichtsmatrix für die Eingabe multipliziert, dazu kommt ein
#    Bias.
# 2. Der bisherige Zustand wird mit einer zweiten Gewichtsmatrix multipliziert, dazu kommt ein
#    zweiter Bias. Vor dem ersten Wort besteht der Zustand nur aus Nullen.
# 3. Beide Ergebnisse werden addiert und durch tanh geschickt. Das ist der neue Zustand.
#
# In PyTorch heißen die Bestandteile so, im Beispiel unten mit 300 Eingabewerten je Wort und
# einem Zustand aus 4 Werten:
#
# | Bestandteil | im Code | Größe im Beispiel |
# |---|---|---|
# | Gewichte für das Wort ("input to hidden") | `rnn.weight_ih_l0` | 4 × 300 |
# | Gewichte für den bisherigen Zustand ("hidden to hidden") | `rnn.weight_hh_l0` | 4 × 4 |
# | die beiden Bias-Vektoren | `rnn.bias_ih_l0`, `rnn.bias_hh_l0` | je 4 |
#
# Das `l0` steht für Schicht 0. Wir rechnen die drei Schritte mit diesen Gewichten von Hand nach:

# %%
torch.manual_seed(0)  # reproduzierbare Startgewichte
# Kleine RNN-Schicht: 300 Eingabewerte je Wort, Hidden State mit nur 4 Werten (gut lesbar)
rnn = nn.RNN(input_size=EMBEDDING_DIM, hidden_size=4, batch_first=True)

satz = "du bist kein Trottel"
x = vectorize(satz).unsqueeze(0)  # (1, 4, 300): Batch aus einem Satz mit 4 Tokens

h = torch.zeros(4)  # Startzustand: noch nichts gelesen
with torch.no_grad():  # nur nachrechnen, keine Gradienten nötig
    for t, token in enumerate(tokenisiere(satz)):
        # weight_ih_l0 (4 × 300) verarbeitet das aktuelle Wort, weight_hh_l0 (4 × 4) den alten Zustand.
        # Es sind in jedem Schleifendurchlauf dieselben beiden Matrizen.
        h = torch.tanh(rnn.weight_ih_l0 @ x[0, t] + rnn.bias_ih_l0 + rnn.weight_hh_l0 @ h + rnn.bias_hh_l0)
        print(f"{token:>8}: {h.numpy().round(3)}")  # Zustand nach diesem Wort

    out, h_n = rnn(x)  # dasselbe in einem Aufruf durch PyTorch
# h_n[0, 0] = Schicht 0, erster (einziger) Satz im Batch; stimmt unsere Schleife mit nn.RNN überein?
print(torch.allclose(h, h_n[0, 0]))

# %% [markdown]
# Jede der vier Zeilen ist der Zustand nach einem Wort: 4 Werte, weil `hidden_size=4`. Das
# `True` in der letzten Zeile bestätigt, dass unsere Schleife nach "Trottel" genau bei dem
# Zustand ankommt, den `nn.RNN` als `h_n` zurückgibt. Die Werte selbst haben noch keine
# Bedeutung, das Netz ist untrainiert. Zwei Dinge fallen trotzdem auf:
#
# - Alle Werte liegen zwischen −1 und 1, dem Wertebereich von tanh, und viele liegen fast genau
#   auf −1 oder 1. Die Summe, die tanh bekommt, ist hier so groß, dass tanh sie nur noch auf
#   seinen Rand abbilden kann; man sagt, tanh ist gesättigt. Der Grund: Für jedes Wort werden
#   300 Eingabewerte mit je einem Gewicht multipliziert und aufaddiert, das ergibt schnell große
#   Zahlen.
# - Der Zustand sieht nach jedem Wort völlig anders aus als nach dem vorherigen.
#
# Beides zusammen wirft eine Frage auf: Wie viel vom Satzanfang steckt im Zustand nach
# "Trottel" überhaupt noch? Wir vergleichen den Endzustand für Sätze, die sich vor "Trottel"
# unterscheiden, und zerlegen die Summe im letzten Schritt in ihre beiden Anteile:

# %%
with torch.no_grad():
    for vergleichssatz in ["du bist kein Trottel", "du bist ein Trottel", "kein Problem , du bist ein Trottel",
                           "Trottel"]:
        _, h_ende = rnn(vectorize(vergleichssatz).unsqueeze(0))
        print(f"{vergleichssatz:>36}: {h_ende[0, 0].numpy().round(3)}")  # Zustand nach dem letzten Wort

    # Letzter Schritt für "du bist kein Trottel" in seine beiden Anteile zerlegt
    _, h_vor_trottel = rnn(vectorize("du bist kein").unsqueeze(0))  # Zustand nach "kein"
    anteil_wort = rnn.weight_ih_l0 @ vectorize("Trottel")[0] + rnn.bias_ih_l0
    anteil_zustand = rnn.weight_hh_l0 @ h_vor_trottel[0, 0] + rnn.bias_hh_l0
print("Anteil des Wortes \"Trottel\":   ", anteil_wort.numpy().round(2))
print("Anteil des bisherigen Zustands:", anteil_zustand.numpy().round(2))

# %% [markdown]
# Alle vier Endzustände sind nahezu gleich, sogar für "Trottel" ganz allein. Ob davor "ein" oder
# "kein" stand, hinterlässt praktisch keine Spur. Die Zerlegung zeigt den Grund: Der Anteil des
# aktuellen Wortes ist um ein Vielfaches größer als der Anteil des bisherigen Zustands, und tanh
# bildet die Summe ohnehin auf fast dieselben Randwerte ab. Ein untrainiertes RNN hat also noch
# kein brauchbares Gedächtnis, im Wesentlichen sieht es nur das letzte Wort.
#
# Die Gewichte so einzustellen, dass frühere Wörter wie "kein" eine Spur im Zustand
# hinterlassen, ist genau die Aufgabe des Trainings. Dass das gelingt, zeigt Abschnitt 7: Das
# trainierte Netz unterscheidet "ein Trottel" und "kein Trottel" deutlich. Die Sättigung von
# tanh begegnet uns in Abschnitt 6 noch einmal, denn wo tanh flach ist, kommt beim
# Rückwärtsrechnen kaum Gradient durch.
#
# Wichtig ist außerdem, was in der Schleife *nicht* passiert: Es gibt nicht vier verschiedene
# Gewichtssätze für vier Positionen. Dieselben Matrizen `weight_ih_l0` und `weight_hh_l0` werden
# in jedem Schritt wiederverwendet. Deshalb kann ein RNN Folgen beliebiger Länge verarbeiten,
# und was es an einer Stelle über ein Wort wie "kein" lernt, gilt an jeder Position im Satz.
#
# Bisher hatten wir eine einzelne Schicht und haben sie von Hand durchgerechnet. In der Aufgabe
# stecken in `nn.GRU` drei solche Schichten übereinander: Schicht 1 liest die Wörter, Schicht 2
# liest die Zustände von Schicht 1, Schicht 3 die von Schicht 2. Jede Schicht arbeitet dabei
# genau wie unsere Schleife, Wort für Wort.
#
# Alle Zustände, die dabei entstehen, kann man sich als Tabelle vorstellen: eine Zeile pro
# Schicht, eine Spalte pro Wort. In jedem Feld steht der Zustand, den diese Schicht nach diesem
# Wort hat. Für unseren Satz mit vier Wörtern:
#
# ```
#           out: Zustände der Schicht 3 an allen Positionen
#           ┌──────────┬──────────┬──────────┐
# Schicht 3 ● ───────► ● ───────► ● ───────► ●  → h_n[2] = h_n[-1]
#           ▲          ▲          ▲          ▲
# Schicht 2 ● ───────► ● ───────► ● ───────► ●  → h_n[1]
#           ▲          ▲          ▲          ▲
# Schicht 1 ● ───────► ● ───────► ● ───────► ●  → h_n[0]
#           ▲          ▲          ▲          ▲
#          "du"      "bist"     "kein"   "Trottel"
# ```
#
# Jeder Punkt ist ein Feld der Tabelle, also ein Zustand mit 8 Werten. Von der ganzen Tabelle
# gibt `nn.GRU` nur zwei Ausschnitte zurück:
#
# - `out` ist die **oberste Zeile**: Schicht 3 nach jedem einzelnen Wort.
# - `h_n` ist die **rechte Spalte**: jede Schicht nach dem letzten Wort.
#
# Für die Klassifikation brauchen wir einen einzigen Vektor pro Satz, und zwar das Feld oben
# rechts: die oberste Schicht nach dem letzten Wort. Dieses Feld liegt in beiden Ausschnitten.
# Das prüfen wir nach:

# %%
torch.manual_seed(0)
gru = nn.GRU(EMBEDDING_DIM, hidden_size=8, num_layers=3, batch_first=True)  # 3 Schichten, Zustand mit 8 Werten

with torch.no_grad():
    out, h_n = gru(x)  # x ist weiterhin "du bist kein Trottel": 1 Satz, 4 Wörter, je 300 Werte

print("out:", tuple(out.shape), "= 1 Satz, 4 Wörter, je 8 Werte")
print("h_n:", tuple(h_n.shape), "= 3 Schichten, 1 Satz, je 8 Werte")

# Feld oben rechts: in out das letzte Wort, in h_n die letzte (oberste) Schicht
print(torch.allclose(out[:, -1], h_n[-1]))

# %% [markdown]
# Die Formen passen zum Bild: `out` hat 4 Einträge, einen pro Wort, `h_n` hat 3 Einträge, einen
# pro Schicht. Und `out[:, -1]` (letztes Wort in der obersten Zeile) ist derselbe Vektor wie
# `h_n[-1]` (oberste Schicht in der rechten Spalte).
#
# Eine Falle gibt es dabei: `batch_first=True` gilt nur für `out`. Bei `h_n` stehen die
# Schichten immer an erster Stelle, deshalb holt `h_n[-1]` die oberste Schicht.
#
# Das LSTM-Beispiel im Buch ist genauso aufgebaut: zwei Schichten übereinander, und für die
# Klassifikation wird die Ausgabe an der letzten Position genommen. Dort steht allerdings die
# Position vorne statt der Batch-Dimension (`batch_first=False`), deshalb heißt es im Buch
# `rnn_o[-1]` statt `out[:, -1]`.
#
# Bei einem einzelnen Satz ist es also egal, welchen der beiden Wege man nimmt. Das ändert sich,
# sobald Tweets unterschiedlicher Länge im selben Batch stecken.

# %% [markdown]
# ## 5. Der letzte Token bei gepaddeten Folgen
#
# Die Gleichheit von `out[:, -1]` und `h_n[-1]` aus Abschnitt 4 gilt für einen einzelnen Satz
# ohne Padding. In einem Batch aus Abschnitt 3 sind aber die meisten Tweets mit Nullvektoren
# aufgefüllt, und "die letzte Position" ist für sie kein Wort mehr.
#
# Abhilfe schafft `pack_padded_sequence`. Die Funktion bekommt den gepaddeten Batch zusammen mit den
# echten Längen (der Liste `lengths` aus Abschnitt 3) und legt intern nur die echten Tokens ab.
# Die GRU verarbeitet dann an jeder Position nur die Tweets, die dort noch
# ein Wort haben; ein Tweet mit 4 Tokens ist nach Schritt 4 fertig. `pad_packed_sequence`
# macht daraus hinterher wieder einen normalen gepaddeten Tensor.
#
# Wir vergleichen für einen kurzen Satz, der zusammen mit einem langen in einem Batch steckt,
# drei Möglichkeiten, seinen letzten Zustand auszulesen, mit dem Ergebnis, das die GRU für den
# kurzen Satz ganz allein liefert:

# %%
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

lang = vectorize("ehrlich , das ist kein Trottel und das weiß jeder")  # 10 Tokens
kurz = vectorize("du bist ein Trottel")  # 4 Tokens
batch = pad_sequence([lang, kurz], batch_first=True)  # (2, 10, 300), "kurz" mit 6 Nullzeilen aufgefüllt
laengen = [lang.size(0), kurz.size(0)]  # [10, 4], entspricht lengths aus collate_batch

with torch.no_grad():
    # Referenz: der kurze Satz allein, ganz ohne Padding. So soll das Ergebnis aussehen.
    _, h_allein = gru(kurz.unsqueeze(0))

    # Ohne Packing: gepaddeten Batch direkt in die GRU, Nullzeilen werden mitverarbeitet
    out_ungepackt, h_ungepackt = gru(batch)

    # Mit Packing: erst packen (mit den echten Längen), dann GRU, dann wieder entpacken.
    # Hier gibt es zwei Stellen zum Auslesen: h_n[-1] und out an der letzten Position.
    gepackt = pack_padded_sequence(batch, laengen, batch_first=True, enforce_sorted=False)
    out_gepackt, h_gepackt = gru(gepackt)
    out_gepackt, _ = pad_packed_sequence(out_gepackt, batch_first=True)

# Jeweils die ersten 4 von 8 Werten des Zustands für den kurzen Satz.
# Indizes: h_n[Schicht, Satz im Batch, Wert], out[Satz im Batch, Position, Wert];
# -1 = oberste Schicht bzw. letzte Position, 1 = der kurze Satz (zweiter im Batch)
print("allein             ", h_allein[-1, 0, :4].numpy().round(4))
print("ohne Packing h_n   ", h_ungepackt[-1, 1, :4].numpy().round(4))
print("gepackt h_n[-1]    ", h_gepackt[-1, 1, :4].numpy().round(4))
print("gepackt out[:, -1] ", out_gepackt[1, -1, :4].numpy().round(4))

# %% [markdown]
# Ohne Packing läuft die GRU nach "Trottel" noch sechs Schritte über Nullvektoren weiter, und
# jeder dieser Schritte verändert den Zustand, obwohl dort gar kein Text mehr steht. Nach
# Abschnitt 4 ist das besonders ungünstig: Ein untrainiertes RNN wird vor allem vom zuletzt
# gelesenen Element geprägt, und das ist hier ein Füllvektor statt "Trottel". Das
# Ergebnis hängt dann davon ab, mit welchen Tweets ein Tweet zufällig im selben Batch landet.
# `pack_padded_sequence` verhindert das: Mithilfe von `lengths` stoppt die Verarbeitung jeder
# Folge an ihrem echten Ende, und `h_n` enthält für jeden Tweet den Zustand nach seinem
# tatsächlich letzten Token. `enforce_sorted=False` erlaubt dabei Batches, die nicht nach Länge
# sortiert sind, und `h_n` kommt trotzdem in der ursprünglichen Reihenfolge zurück.
#
# Die letzte Zeile zeigt die eigentliche Falle für Aufgabe 1.6: Nach `pad_packed_sequence` ist
# `out` an den aufgefüllten Positionen wieder mit Nullen besetzt. `out[:, -1]` liefert für jeden
# Tweet, der kürzer ist als der längste im Batch, einen Nullvektor. Der Hinweis "Ausgabe zum
# letzten Token" in der Aufgabe meint also den letzten *echten* Token jedes Tweets. Den bekommt
# man direkt aus `h_n[-1]`, oder gleichwertig, indem man `out` an Position `Länge − 1`
# ausliest:

# %%
letzte_position = torch.tensor(laengen) - 1  # Index des letzten echten Tokens je Satz: [9, 3]
# Für Satz 0 Position 9, für Satz 1 Position 3 aus out herausgreifen
aus_out = out_gepackt[torch.arange(len(laengen)), letzte_position]
print(torch.allclose(aus_out, h_gepackt[-1]))  # identisch mit h_n[-1]?

# %% [markdown]
# Merksatz: Bei gepackten Folgen ist `h_n[-1]` die Zusammenfassung jedes Tweets, `out[:, -1]`
# nur für den längsten. Bei einem bidirektionalen Netz wäre `h_n[-1]` übrigens die
# Rückwärtsrichtung der obersten Schicht; mit `bidirectional=False` wie in der Aufgabe stellt
# sich diese Frage nicht.

# %% [markdown]
# ## 6. LSTM und GRU: Gates und Parameterzahl
#
# Abschnitt 5 hat geklärt, *welcher* Zustand am Ende ausgelesen wird. Offen ist, wie viel
# dieser Zustand vom Anfang des Tweets überhaupt noch weiß. In Abschnitt 4 hat das untrainierte
# RNN praktisch nur das letzte Wort behalten. Durch Training kann es lernen, sich mehr zu
# merken, aber nur begrenzt.
#
# Wie weit ein Netz manchmal zurückblicken muss, zeigt das Beispiel aus den Folien (ein ganz
# ähnliches steht im Buch): "Ich bin in China geboren und erst mit 14 Jahren nach Deutschland
# gekommen. Ich spreche fließend ...". Das passende nächste Wort, "Chinesisch", hängt an
# "China", und dazwischen liegen dreizehn Tokens.
#
# **Das Problem beim einfachen RNN.** In der Schleife aus Abschnitt 4 wird der Zustand bei jedem
# Wort komplett neu berechnet: Der alte Zustand wird mit den Gewichten verrechnet und durch tanh
# geschickt, das Ergebnis ist der neue Zustand. Die Information "China" muss diese Umrechnung
# also dreizehnmal überstehen.
#
# Dasselbe Problem gibt es beim Lernen. Im Training wird nach jedem Batch für jedes Gewicht
# berechnet, in welche Richtung es verändert werden soll; das ist der Gradient aus
# `loss.backward()`, den Sie aus Termin 1 kennen. Damit das Netz lernen kann, auf "China" zu
# achten, muss dieses Signal vom Ende des Satzes Schritt für Schritt bis zu "China"
# zurücklaufen. Bei jedem Schritt wird es dabei mit einem Faktor multipliziert. Liegt dieser
# Faktor zum Beispiel bei 0,5, ist nach dreizehn Schritten nur noch etwa ein Zehntausendstel
# übrig. Weit zurückliegende Wörter haben dann kaum Einfluss darauf, was das Netz lernt. Das
# nennt man Vanishing Gradient (verschwindender Gradient). Sind die Faktoren größer als 1,
# wächst das Signal stattdessen unkontrolliert an (Exploding Gradient). Ein gesättigtes tanh wie
# in Abschnitt 4 verschärft das Problem: Wo tanh fast flach ist, ist auch der Faktor fast 0.
#
# **Die Idee des LSTM.** Das LSTM führt neben dem Hidden State einen zweiten Zustand mit, den
# Cell State. Die Folien nennen ihn das "Gedächtnis" des LSTM. Anders als der Hidden State wird
# er nicht bei jedem Wort neu berechnet, sondern nur bearbeitet: Ein Teil wird vergessen, etwas
# Neues kommt hinzu, der Rest bleibt einfach stehen.
#
# Auch unsere Beispielsätze brauchen so ein Gedächtnis. Bei "du bist kein Trottel" muss sich das
# Netz bei "kein" eine Verneinung merken und sie bis "Trottel" behalten. Bei "kein Problem , du
# bist ein Trottel" muss es die Verneinung nach "Problem" wieder vergessen, sonst würde es den
# Satz fälschlich für harmlos halten. Genau diese Unterscheidung muss das Modell in Abschnitt 7
# lernen.
#
# Was vergessen, gespeichert und ausgegeben wird, steuern drei sogenannte Gates (englisch für
# "Tore"). Ein Gate ist eine kleine Schicht, die aus dem aktuellen Wort und dem bisherigen Hidden
# State für jeden Eintrag des Gedächtnisses eine Zahl zwischen 0 und 1 berechnet. Dafür sorgt
# die Sigmoid-Funktion am Ausgang, die jede Zahl in diesen Bereich drückt. 0 heißt "sperren",
# 1 heißt "durchlassen", Werte dazwischen lassen einen Teil durch. Die Gewichte der Gates werden
# wie alle anderen trainiert; das Netz lernt also selbst, wann es sich etwas merken soll.
#
# - Das **Forget Gate** legt fest, wie viel von jedem Eintrag im Gedächtnis erhalten bleibt.
# - Das **Input Gate** legt fest, wie viel von den neuen Werten, die aus dem aktuellen Wort
#   berechnet werden, ins Gedächtnis aufgenommen wird.
# - Das **Output Gate** legt fest, wie viel vom Gedächtnis als Hidden State nach außen gegeben
#   wird, also an die nächste Schicht oder am Ende an die lineare Schicht.
#
# Warum hilft das? Im Gedächtnis wird ein Eintrag von Wort zu Wort nur mit dem Wert des Forget
# Gates multipliziert, ohne Gewichtsmatrix und ohne tanh dazwischen. Die Folien sprechen deshalb
# von einem "linearen Gedächtnis". Lernt das Netz für einen Eintrag ein Forget Gate von 0,95, ist
# nach dreizehn Schritten noch rund die Hälfte davon übrig statt eines Zehntausendstels wie im
# Beispiel oben. Der Gradient läuft im Training auf demselben Weg zurück und bleibt deshalb
# ebenfalls erhalten.
#
# **Die GRU** setzt dieselbe Idee schlanker um. Sie hat kein eigenes Gedächtnis, sondern
# bearbeitet direkt den Hidden State, und kommt mit zwei Gates aus. Das **Update Gate** legt je
# Eintrag fest, wie viel vom alten Zustand erhalten bleibt und wie viel durch einen neu
# berechneten Wert ersetzt wird; es übernimmt also die Aufgaben von Forget und Input Gate
# zusammen. Das **Reset Gate** legt fest, wie stark der alte Zustand in diesen neuen Wert
# einfließt.
#
# In der Aufgabe verwenden Sie eine GRU, das Modell in Abschnitt 7 dieses Notebooks ist ein
# LSTM. Beide beruhen auf derselben Grundidee: Information wird nicht bei jedem Wort neu
# berechnet, sondern gezielt behalten oder ersetzt.
#
# Diese Gates sind in PyTorch nicht versteckt, sondern stecken als übereinander gestapelte Blöcke
# in den Gewichtsmatrizen:

# %%
hidden = 16
# Je eine Schicht LSTM und GRU mit gleicher Eingabe- und Zustandsgröße
lstm = nn.LSTM(EMBEDDING_DIM, hidden, batch_first=True)
gru_1 = nn.GRU(EMBEDDING_DIM, hidden, batch_first=True)

# weight_ih_l0 = Gewichte für die Eingabe (input -> hidden) in Schicht 0, alle Gates übereinander
print(lstm.weight_ih_l0.shape)  # 4 Blöcke à 16 Zeilen: Input, Forget, Kandidat, Output
print(gru_1.weight_ih_l0.shape)  # 3 Blöcke à 16 Zeilen: Reset, Update, Kandidat

with torch.no_grad():
    _, lstm_zustand = lstm(x)  # zweiter Rückgabewert: beim LSTM ein Tupel
    _, gru_zustand = gru_1(x)  # bei der GRU ein einzelner Tensor
print(type(lstm_zustand), len(lstm_zustand))
print(type(gru_zustand))

# %% [markdown]
# Die erste Dimension ist 4 × 16 beim LSTM und 3 × 16 bei der GRU, ein Block je Gate bzw.
# Kandidatenschicht. Außerdem unterscheiden sich die Rückgabewerte: Das LSTM liefert als
# zweiten Wert ein Tupel `(h_n, c_n)`, die GRU nur `h_n`. Wer in der Aufgabe statt der GRU ein
# LSTM ausprobiert, muss deshalb auch den Aufruf anpassen und das Tupel entpacken. Ausgelesen
# wird in beiden Fällen der letzte Zustand der obersten Schicht aus `h_n`, der Cell State
# `c_n` bleibt intern.
#
# Aus der Blockstruktur folgt direkt die Parameterzahl. Jeder Block hat dieselben Bestandteile
# wie die RNN-Zelle aus Abschnitt 4: eine Gewichtsmatrix für die Eingabe, eine für den alten
# Zustand und zwei Bias-Vektoren. Für die erste LSTM-Schicht oben mit 16 Zustandswerten und 300
# Eingabewerten sind das je Block 16 × 300 + 16 × 16 + 2 × 16 = 5.088 Werte, bei 4 Blöcken also
# 20.352. Jede weitere Schicht bekommt nicht die Wortvektoren, sondern den Zustand der Schicht
# darunter, also nur 16 statt 300 Eingabewerte, und ist entsprechend kleiner. Die GRU zählt
# genauso, nur mit 3 statt 4 Blöcken. (Das gilt für ein nicht bidirektionales Netz wie in der
# Aufgabe.) Wir prüfen diese Zählweise an Netzen mit zwei Schichten:

# %%
# Zählt die Einträge aller Parameter-Tensoren, die vom Optimizer verändert werden
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Die Zählweise aus dem Text, getrennt nach erster und weiteren Schichten
def von_hand_gezaehlt(bloecke, eingabe, hidden, schichten):
    # erste Schicht: Eingabe sind die 300-dimensionalen Wortvektoren
    erste = bloecke * (hidden * (eingabe + hidden) + 2 * hidden)
    # weitere Schichten: Eingabe ist der Zustand der Schicht darunter, also hidden
    weitere = bloecke * (hidden * (hidden + hidden) + 2 * hidden)
    return erste + (schichten - 1) * weitere


lstm_2 = nn.LSTM(EMBEDDING_DIM, hidden, num_layers=2)
gru_2 = nn.GRU(EMBEDDING_DIM, hidden, num_layers=2)

# Links von PyTorch gezählt, rechts von Hand
print("LSTM:", count_parameters(lstm_2), von_hand_gezaehlt(4, EMBEDDING_DIM, hidden, 2))
print("GRU: ", count_parameters(gru_2), von_hand_gezaehlt(3, EMBEDDING_DIM, hidden, 2))

# %% [markdown]
# Beide Zählungen stimmen überein. Die GRU hat bei gleicher Größe genau drei Viertel der
# Parameter eines LSTM, der größte Anteil entfällt auf die erste Schicht, weil dort die 300
# Eingabedimensionen eingehen. Für die Wahl zwischen beiden bei einem kleinen Datensatz wie
# GermEval heißt das: Weniger Parameter bedeuten schnelleres Training und weniger Spielraum zum
# Auswendiglernen; das LSTM kann dafür bei langen Folgen im Vorteil sein. Eine pauschal bessere
# Wahl gibt es nicht, deshalb lohnt sich der Vergleich in der Aufgabe. Die lineare Schicht am
# Ende fällt mit hidden × 2 + 2 Parametern kaum ins Gewicht.

# %% [markdown]
# ## 7. Training und Auswertung
#
# Mit gepackten Batches, dem richtigen letzten Zustand und der Wahl der Zelle aus den Abschnitten
# 3 bis 6 lässt sich jetzt ein vollständiges Modell trainieren. Ziel ist, für jeden künstlichen
# Satz aus Abschnitt 3 vorherzusagen, ob er beleidigend ist (`OFFENSE`, Label 1) oder nicht
# (`OTHER`, Label 0). Wir verwenden hier die LSTM-Variante; die Struktur entspricht dem Netz aus
# der Aufgabe, nur mit `nn.LSTM` statt `nn.GRU`.

# %%
class TextRNN(nn.Module):
    def __init__(self, hidden_dim, embedding_dim=EMBEDDING_DIM, dropout=0.4):
        super().__init__()
        # dropout wirkt zwischen den beiden LSTM-Schichten, deshalb num_layers > 1
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 2)  # letzter Zustand -> 2 Klassen (OTHER, OFFENSE)

    def forward(self, _x):
        x, input_lengths = _x  # wie in der Aufgabe: Tupel aus gepaddetem Batch und echten Längen
        # Packen, damit das LSTM jede Folge an ihrem echten Ende stoppt (Abschnitt 5)
        x = pack_padded_sequence(x, input_lengths, batch_first=True, enforce_sorted=False)
        x, (ht, ct) = self.lstm(x)  # LSTM: zweiter Rückgabewert ist (h_n, c_n)
        return self.linear(ht[-1])  # letzter echter Zustand der obersten Schicht, Abschnitt 5

# %% [markdown]
# Bevor trainiert wird, ein Detail, das in der Trainingsschleife der Aufgabe leicht übersehen
# wird. Das Modell enthält Dropout, und Dropout verhält sich im Trainings- und im
# Auswertungsmodus unterschiedlich:

# %%
torch.manual_seed(0)
model = TextRNN(hidden_dim=16, dropout=0.5)  # untrainiert, hohes Dropout macht den Effekt deutlich
labels, texts, lengths = next(iter(demo_loader))

with torch.no_grad():
    model.train()  # Trainingsmodus: Dropout aktiv
    # Zweimal dieselbe Eingabe, ausgegeben werden die beiden Logits des ersten Tweets
    print(model((texts, lengths))[0], model((texts, lengths))[0])
    model.eval()  # Auswertungsmodus: Dropout aus
    print(model((texts, lengths))[0], model((texts, lengths))[0])

# %% [markdown]
# Im Trainingsmodus liefert dieselbe Eingabe bei jedem Aufruf ein anderes Ergebnis, weil Dropout
# zufällig Werte zwischen den Schichten abschaltet. Im Auswertungsmodus ist das Ergebnis
# reproduzierbar. Die Schleife in der Aufgabe ruft `model.eval()` nie auf; `torch.no_grad()`
# allein schaltet Dropout nicht ab, es verhindert nur den Aufbau des Berechnungsgraphen. Die
# Test-Accuracy wird dort also mit aktivem Dropout gemessen und fällt etwas schlechter und
# unruhiger aus, als das Modell eigentlich ist. Die folgende Trainingsfunktion setzt deshalb vor
# jeder Phase den passenden Modus, wie in Termin 2.
#
# Für den Fortschrittsbalken verwenden wir `from tqdm import tqdm` statt `tqdm.notebook`. Die
# Notebook-Variante zeichnet ihren Balken als Widget per JavaScript und bleibt auf Servern ohne
# Widget-Unterstützung leer; die einfache Variante schreibt einen Textbalken, der überall
# funktioniert.

# %%
from tqdm import tqdm

# Gleiche Bauweise wie in der Aufgabe; Testdaten müssen nicht gemischt werden
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=64, collate_fn=collate_batch)


# Anteil richtig klassifizierter Sätze in einem Datensatz
def accuracy(model, loader):
    model.eval()  # Dropout aus
    richtige = 0
    with torch.no_grad():  # kein Berechnungsgraph nötig
        for target, text, length in loader:
            # argmax über die 2 Logits = vorhergesagte Klasse; mit dem Label vergleichen und zählen
            richtige += (model((text, length)).argmax(dim=1) == target).sum().item()
    return richtige / len(loader.dataset)


def trainiere(model, epochs, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # derselbe Optimizer wie in der Aufgabe
    loss_fn = nn.CrossEntropyLoss()  # erwartet rohe Logits, passend zu self.linear ohne Aktivierung
    train_verlauf, test_verlauf = [], []
    for _ in tqdm(range(epochs)):  # Textbalken statt Widget, funktioniert ohne JavaScript
        model.train()  # Dropout an
        for target, text, length in train_dataloader:
            optimizer.zero_grad()  # Gradienten aus dem letzten Batch löschen
            loss = loss_fn(model((text, length)), target)
            loss.backward()  # Gradienten berechnen
            optimizer.step()  # Gewichte anpassen
        # Nach jeder Epoche auf beiden Datensätzen messen (accuracy schaltet selbst auf eval)
        train_verlauf.append(accuracy(model, train_dataloader))
        test_verlauf.append(accuracy(model, test_dataloader))
    return train_verlauf, test_verlauf


torch.manual_seed(0)  # reproduzierbare Startgewichte und Batch-Reihenfolge
model = TextRNN(hidden_dim=16, dropout=0.2)
train_verlauf, test_verlauf = trainiere(model, epochs=15, lr=1e-3)
print("Training:", [round(a, 2) for a in train_verlauf])
print("Test:    ", [round(a, 2) for a in test_verlauf])

# %% [markdown]
# Nach der ersten Epoche liegen beide Werte genau beim Anteil der `OTHER`-Sätze im jeweiligen
# Datensatz: Das Modell sagt zunächst für jeden Satz die häufigere Klasse voraus. Danach steigt
# die Test-Accuracy deutlich an. Die Trainings-Accuracy bleibt dagegen bei etwa 85 % stehen. Das
# ist kein Fehler, sondern genau das gewünschte Verhalten: Bei 15 % vertauschten Labels trifft
# ein Modell, das die Regel gelernt hat, auf den Trainingsdaten nur etwa 85 %. Mehr ginge nur,
# wenn es zusätzlich die vertauschten Labels auswendig lernt. Ob das Modell tatsächlich gelernt
# hat, worauf es ankommt, zeigen einzelne Sätze, die sich nur in einem Wort oder in der
# Reihenfolge unterscheiden:

# %%
# Wahrscheinlichkeit für OFFENSE bei einem einzelnen, frei formulierten Satz
def vorhersage(model, text):
    model.eval()
    with torch.no_grad():
        vektoren = vectorize(text).unsqueeze(0)  # Batch aus einem Satz: (1, Anzahl Tokens, 300)
        logits = model((vektoren, [vektoren.size(1)]))  # Länge = alle Tokens, kein Padding
        # softmax macht aus den Logits Wahrscheinlichkeiten; [0, 1] = erster Satz, Klasse OFFENSE
        wahrscheinlichkeit = F.softmax(logits, dim=1)[0, 1]
    return wahrscheinlichkeit.item()


# Paare, die sich nur in "ein"/"kein" oder in der Reihenfolge unterscheiden; "Idiot" kam im Training nie vor
for text in ["du bist ein Trottel", "du bist kein Trottel", "kein Problem , du bist ein Trottel",
             "mein Nachbar ist ein Profi !", "sie ist wirklich ein Idiot heute wieder",
             "sie ist wirklich kein Idiot"]:
    print(f"{vorhersage(model, text):.2f}  {text}")

# %% [markdown]
# Ausgegeben ist die vorhergesagte Wahrscheinlichkeit für `OFFENSE`. "kein Trottel" wird als
# harmlos erkannt, "kein Problem , ... ein Trottel" dagegen als beleidigend, obwohl beide Sätze
# "kein" und "Trottel" enthalten. Das Modell hat also die Reihenfolge ausgewertet, nicht nur das
# Vorkommen einzelner Wörter. Genau das konnte das untrainierte RNN in Abschnitt 4 noch nicht:
# Das Training hat die Gewichte so eingestellt, dass "kein" direkt vor dem Schimpfwort eine Spur
# im Zustand hinterlässt. Auch "Idiot" wird richtig eingeordnet, obwohl das Wort im
# Training nie vorkam: Sein Vektor liegt nahe bei den bekannten Schimpfwörtern, und genau
# darauf beruht der Nutzen vortrainierter Embeddings.
#
# Einzelne Sätze zeigen, *was* das Modell gelernt hat. Wie gut es insgesamt ist, misst die
# Accuracy: der Anteil der Testsätze, die es richtig einordnet. Eine Accuracy von 90 % klingt
# gut, ist für sich allein aber wenig aussagekräftig. Man muss wissen, was man ohne jedes Lernen
# schon erreicht hätte.
#
# Der einfachste Vergleich ist ein "Modell", das den Text gar nicht ansieht und immer die
# häufigere Klasse vorhersagt, hier also immer `OTHER`. Es liegt bei jedem `OTHER`-Satz richtig
# und bei jedem `OFFENSE`-Satz falsch. Seine Accuracy ist damit genau der Anteil der
# `OTHER`-Sätze im Testset. Bei unseren 300 Testsätzen sind 208 davon `OTHER`; dieses "Modell"
# käme also auf knapp 70 %, ohne ein einziges Wort gelesen zu haben.
#
# Diese Zahl ist die Untergrenze. Was ein trainiertes Modell darüber hinaus schafft, hat es
# tatsächlich aus den Texten gelernt. Liegt seine Accuracy nur knapp darüber, hat es kaum mehr
# gelernt, als dass `OTHER` häufiger vorkommt. Der Vergleich in Zahlen:

# %%
# Accuracy eines "Modells", das immer OTHER sagt = Anteil der OTHER-Sätze im Testset
anteil_other = sum(r.primary_label == "OTHER" for r in test_data) / len(test_data)
print(f"Immer OTHER:  {anteil_other:.1%}")
print(f"TextRNN:      {test_verlauf[-1]:.1%}")

# %% [markdown]
# Auf den künstlichen Sätzen liegt das Modell klar über dieser Untergrenze. Bei GermEval liegt
# sie ähnlich hoch: Im Testset sind 2.330 von 3.532 Tweets `OTHER`, also 66 %. Eine Accuracy
# um 70 % ist dort kaum mehr als das Raten der häufigeren Klasse; diesen Maßstab sollten Sie
# bei der Frage nach der besten erreichten Accuracy im Kopf haben.
#
# Aufgabe 1.8 verlangt außerdem, `hidden_dim`, `dropout` und die Lernrate zu variieren. Statt
# eines interaktiven Reglers trainieren wir einige Kombinationen nacheinander und stellen die
# Verläufe nebeneinander, links auf den Test-, rechts auf den Trainingsdaten. Die Liste
# `varianten` ist die Stelle zum Ändern; nach jeder Änderung die Zelle erneut ausführen. Der
# Durchlauf dauert auf einer CPU etwa eine Minute.

# %%
import matplotlib.pyplot as plt

# Hier ändern: jede Zeile ist eine Kombination, die 30 Epochen trainiert wird
varianten = [
    {"hidden_dim": 16, "dropout": 0.2, "lr": 1e-3},  # Ausgangsmodell von oben
    {"hidden_dim": 4, "dropout": 0.2, "lr": 1e-3},  # kleinerer Hidden State
    {"hidden_dim": 16, "dropout": 0.2, "lr": 1e-4},  # zehnmal kleinere Lernrate
    {"hidden_dim": 128, "dropout": 0.0, "lr": 5e-3},  # großes Netz, kein Dropout, hohe Lernrate
]

# Zwei Diagramme nebeneinander mit gemeinsamer y-Achse
fig, (links, rechts) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for variante in varianten:
    torch.manual_seed(0)  # gleiche Startgewichte und Batch-Reihenfolge für einen fairen Vergleich
    modell = TextRNN(hidden_dim=variante["hidden_dim"], dropout=variante["dropout"])
    train_kurve, test_kurve = trainiere(modell, epochs=30, lr=variante["lr"])
    epochen = range(1, len(test_kurve) + 1)
    links.plot(epochen, test_kurve, label=str(variante))
    rechts.plot(epochen, train_kurve)  # gleiche Farbreihenfolge wie links, daher keine zweite Legende

# Waagerechte Bezugslinien
links.axhline(anteil_other, color="gray", linestyle="--", label="immer OTHER")
rechts.axhline(0.85, color="gray", linestyle=":", label="Obergrenze bei 15 % Labelfehlern")
links.set(title="Testdaten", xlabel="Epoche", ylabel="Accuracy")
rechts.set(title="Trainingsdaten", xlabel="Epoche")
links.legend(fontsize=8)
rechts.legend(fontsize=8)
plt.show()

# %% [markdown]
# Links ist die Test-Accuracy je Epoche für jede Variante zu sehen, mit der Untergrenze
# "immer OTHER" als gestrichelter Linie; rechts die Accuracy auf den Trainingsdaten mit der
# Obergrenze, die bei 15 % vertauschten Labels für ein regeltreues Modell gilt. Drei
# Beobachtungen lassen sich ablesen:
#
# - Mit nur 4 Zustandsdimensionen lernt das Netz die Regel ebenfalls, braucht dafür aber mehr
#   Epochen als mit 16.
# - Die zehnmal kleinere Lernrate 1e-4 klebt lange an der Untergrenze und erreicht die Regel
#   in 30 Epochen nicht vollständig. Eine zu kleine Lernrate sieht in wenigen Epochen aus wie
#   ein Modell, das nichts lernt.
# - Das große Netz mit `hidden_dim=128`, ohne Dropout und mit hoher Lernrate, ist anfangs am
#   schnellsten. Danach steigt seine Trainings-Accuracy über die 85 %-Linie, es lernt also die
#   vertauschten Labels auswendig, und gleichzeitig fällt die Test-Accuracy deutlich ab. Das ist
#   Überanpassung in Reinform.
#
# Für die Aufgabe folgt daraus: Die beste Test-Accuracy liegt nicht unbedingt in der letzten
# Epoche, und eine steigende Trainings-Accuracy ist allein kein gutes Zeichen. Vergleichen Sie
# Trainings- und Testkurven immer gemeinsam.

# %% [markdown]
# ## 8. Zusammenfassung
#
# - Tokenisierung: spaCy trennt Satzzeichen als eigene Tokens ab, deshalb ergeben 6 Wörter mit
#   Satzzeichen 8 Vektoren. One-Hot-Vektoren sind lang, dünn besetzt und kennen keine
#   Ähnlichkeit; Word Embeddings sind dicht (bei spaCy 300 Dimensionen) und bilden ähnliche
#   Wörter auf benachbarte Vektoren ab.
# - Vortrainierte Embeddings: im Modell als `nn.Embedding.from_pretrained(..., freeze=True)`
#   oder wie in der Aufgabe vor dem Modell in `collate_batch`. In beiden Fällen werden sie
#   nicht trainiert und zählen nicht zu den trainierbaren Parametern.
# - Padding: `pad_sequence` füllt nur bis zum längsten Tweet des Batches auf, deshalb schwankt
#   die mittlere Batch-Dimension; `lengths` merkt sich die echten Längen.
# - RNN: dieselben Gewichte an jeder Position, der Hidden State trägt den bisherigen Text.
#   Untrainiert behält es kaum mehr als das letzte Wort; ein Gedächtnis für frühere Wörter
#   entsteht erst durch Training. `out` hat die Form (Batch, Position, hidden), `h_n` die Form
#   (Schichten, Batch, hidden), unabhängig von `batch_first`.
# - Letzter Token: mit `pack_padded_sequence` ist `h_n[-1]` der Zustand nach dem letzten echten
#   Token jedes Tweets; `out[:, -1]` nach `pad_packed_sequence` ist für kürzere Tweets ein
#   Nullvektor.
# - LSTM und GRU: Gates steuern, was behalten wird, und stabilisieren so den Gradientenfluss.
#   LSTM hat 4 Gewichtsblöcke und gibt `(h_n, c_n)` zurück, GRU hat 3 Blöcke und gibt nur `h_n`
#   zurück. Je Block zählen eine Eingabe- und eine Zustandsmatrix plus zwei Bias-Vektoren.
# - Training: `model.train()` vor dem Training, `model.eval()` und `torch.no_grad()` vor der
#   Auswertung, sonst bleibt Dropout aktiv. Die Accuracy immer gegen die Untergrenze "immer
#   die häufigere Klasse" einordnen. `from tqdm import tqdm` funktioniert auch ohne
#   Widget-Unterstützung.
# - Hyperparameter: Trainings- und Testkurve immer gemeinsam ansehen. Eine zu kleine Lernrate
#   sieht anfangs aus wie ein Modell, das nichts lernt; steigt die Trainings-Accuracy weiter,
#   während die Test-Accuracy fällt, lernt das Modell die Trainingsdaten auswendig.
