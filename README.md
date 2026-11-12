# AI Visual Lab — Projekt TPV

Interaktywna aplikacja edukacyjna dla trzyosobowego zespołu. Backend i obliczenia: Python, Flask, NumPy. Frontend: HTML/CSS i JavaScript Canvas, bez Node.js i budowania bundla. K-Means jest zaimplementowany samodzielnie, bez scikit-learn.

## Uruchomienie (Python 3.11+)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Aplikacja uruchamia serwer produkcyjny Waitress. Otwórz http://127.0.0.1:5000. Jeśli PowerShell blokuje aktywację, używaj `.venv\Scripts\python.exe` zamiast `python`, bez aktywowania środowiska. Na Linux/macOS aktywacja: `source .venv/bin/activate`.

## Zakres

- Strona startowa, nawigacja, responsywny interfejs, instrukcje po polsku.
- Generator 1–2000 punktów 2D, liczba skupisk, szum jako odchylenie standardowe, seed.
- Edycja punktów kliknięciem (dodawanie lub usuwanie najbliższego punktu w promieniu 14 pikseli).
- Zapis wielu nazwanych zbiorów w localStorage przeglądarki, wczytywanie, usuwanie oraz import/eksport JSON.
- K-Means: inicjalizacja k-means++, 1–12 klastrów, maks. 200 iteracji, deterministyczny seed; historia przypisań, centroidów i inertia.
- Start, pauza, wznowienie, pojedynczy krok, reset, tempo, suwak iteracji, trajektorie centroidów i historia inertia.
- Porównanie dwóch wartości k na dokładnie tych samych danych i z tym samym seed, obok siebie (na wąskich ekranach pionowo). Dwa przebiegi mają wspólną oś iteracji; krótszy pozostaje na końcowym stanie.
- Konfiguracja CI sprawdzająca zależności i składnię plików Python, bez testów automatycznych.

KNN i Gradient Descent mają strony integracyjne, bez pozorowania działających algorytmów. Ich implementacja należy do pozostałych osób w zespole. Hosting publiczny wymaga wdrożenia we własnym koncie; repozytorium zawiera instrukcję, nie aktywną usługę.

## Architektura

```text
app.py                      uruchomienie serwera Waitress
lab/__init__.py             fabryka Flask i endpointy
lab/datasets.py             wspólny generator
lab/algorithms/kmeans.py    czyste funkcje NumPy, bez Flask
lab/templates/index.html   wspólna powłoka i widoki modułów
lab/static/app.js          stan UI, wykresy Canvas, sterowanie animacją
lab/static/style.css       wspólny wygląd i responsywność
```

Backend oblicza kompletną historię przebiegu; przeglądarka odtwarza jej klatki. Pauza zatrzymuje odtwarzanie, a nie obliczenia serwera. Każda klatka zawiera centroidy, najbliższe im przypisania oraz zgodną z nimi inertia. Iteracja 0 to inicjalizacja. Pusty klaster zachowuje poprzedni centroid. Zbieżność: największe przesunięcie centroidu ≤ 1e-7. Osiągnięcie limitu nie oznacza zbieżności. Maksymalna liczba klastrów nie może przekroczyć liczby unikalnych punktów.

Wszystkie dane eksperymentu są przesyłane w żądaniu, serwer nie przechowuje danych użytkowników. Zapisane zbiory pozostają w danej przeglądarce i domenie; czyszczenie danych witryny je usuwa. JSON służy do przenoszenia danych. Eksport zawiera dane, nie konfigurację eksperymentu.

### Integracja KNN i Gradient Descent

1. Dodaj czysty moduł `lab/algorithms/knn.py` lub `gradient_descent.py` z obliczeniami niezależnymi od HTTP.
2. Dodaj osobny Flask `Blueprint` w pliku `lab/knn_routes.py` / `lab/gradient_routes.py` i zarejestruj go w `create_app()` przez `app.register_blueprint(...)`. Endpointy umieść pod `/api/knn` i `/api/gradient`.
3. Zastąp sekcję `#knn` lub `#gradient` w szablonie swoim interfejsem. Logikę frontendową dodaj w osobnym pliku JS, aby ograniczyć konflikty ze wspólnym `app.js`. Nie używaj globalnych nazw zmiennych (IIFE lub moduł ES).
4. Wspólny format zbioru: `{"version":1,"points":[[x,y],...]}`. Współrzędne: skończone liczby w [-1000,1000]. KNN potrzebuje też `labels` i osobnej walidacji zgodności etykiet z punktami. Gradient Descent może wymagać innego wejścia, zależnie od wybranej funkcji celu.
5. Użyj generatora przez `POST /api/datasets/generate` lub zaimportuj wyeksportowany JSON.

### API

`POST /api/datasets/generate`

```json
{"n":200,"clusters":3,"noise":5,"seed":42}
```

Odpowiedź: `{"points":[[x,y],...],"seed":42}`. Skupiska gaussowskie są przycinane do [0,100]. Przy szumie 0 punkty mogą się powtarzać, przez co dopuszczalne k jest mniejsze.

`POST /api/kmeans`

```json
{"points":[[0,0],[1,1],[90,90]],"k":2,"seed":42,"max_iterations":100}
```

Odpowiedź: `{"history":[{"iteration":0,"centroids":[...],"labels":[...],"inertia":0.0},...],"converged":true,"k":2,"seed":42}`.

Niepoprawne dane: HTTP 400 z `{"error":"opis"}`. Limit ciała żądania 512 KB: HTTP 413. Obliczenia są synchroniczne; limity punktów/klastrów/iteracji ograniczają koszt pojedynczego żądania. Publiczna usługa dla dużego ruchu wymaga osobnego rate limitingu i limitu współbieżności.

## Weryfikacja

```powershell
python -m pip check
python -m compileall -q app.py lab
```

CI sprawdza zgodność zainstalowanych zależności oraz składnię plików Python. Projekt obecnie nie zawiera testów automatycznych.

Manualnie: wygeneruj dane, uruchom i zatrzymaj animację, przejdź po suwaku, włącz porównanie, edytuj punkt, zapisz i wczytaj zbiór, wykonaj eksport/import. Sprawdź szerokość telefonu i obsługę klawiaturą. Canvas ma opis alternatywny, ale pełny dostęp do informacji o każdym punkcie przez czytnik ekranu nie jest zaimplementowany.

## Wdrożenie

Na hostingu obsługującym Python (np. Render): dodaj repozytorium jako Web Service, ustaw instalację `pip install -r requirements.txt` i polecenie startowe:

```sh
waitress-serve --listen=0.0.0.0:$PORT --call lab:create_app
```

Alternatywne uruchomienie serwera Waitress na porcie 8080, także Windows:

```powershell
waitress-serve --listen=127.0.0.1:8080 --call lab:create_app
```

`python app.py` uruchamia Waitress pod adresem `127.0.0.1:5000`, bez debuggera i automatycznego przeładowywania. Adres `0.0.0.0` w poleceniu hostingowym udostępnia serwer na wszystkich interfejsach. Wdrożenie wymaga konta i konfiguracji domeny/HTTPS po stronie dostawcy. Aplikacja nie wymaga bazy danych ani kluczy API. Font Google jest opcjonalny — bez sieci działa font systemowy.
