# HuaweiTecharena24
Repository for HuaweiTecharena24

# Kunskapsbas
## ANN
Approximate Nearest Neighbors (ANN) to problem przyśpieszania algosa KNN z możliwie najmniejszą utratą dokładności. Wyobraźmy sobie poniższą sytuację:
Mamy portal społecznościowy w którym chcielibyśmy polecać proponowanych znajomych robiąc z ludzi wektory i używając na nich KNN
Robimy to i na początku działa spoko, ale potem Elon Musk pisze o naszym serwisie społecznościowym i mamy nagle 10mln ludzi na serwisie
KNN za każdym razem musi przeszukać te 10mln użytkowników i sprawdzić którzy są blisko co jest mało efektywne jeśli chcielibyśmy to robić co odświeżenie strony
Tak powstało ANN - przyspieszenie KNN z jednoczesnym możliwie niskim spadkiem dokładności.

### Pomysł pierwszy - zamiast exhaustive search zrobić grid trick
Exhaustive search polega na porównaniu każdego punktu z naszym co zajmuje liniowy czas.

Tutaj pojawia się grid trick - dzielimy najpierw przestrzeń na grid składający sie np z 25 kwadratów potem sprawdzamy do którego kwadratu należy nasz punkt - mamy wtedy (przypadek optymistyczny) n/25 przeszukań.
Można potem dzielić kolejne kwadraty na znowu 25 kwadratów przez co mamy n / (25^2) itd. Koniec końców schodzimy do log(n) (całkiem szybcior)

Ale tak na prawdę to czym się różni to od binsearcha - dzielimy przestrzeń na 2 mamy n/2, podprzestrzenie dzielimy na 2 i mamy n/4. 
Dzielimy na 25 tylko wtedy gdy mamy więcej niż 25 punktów (inaczej dojdziemy gdzieś do pustego kwadratu a tego nie chcemy). 
Tak samo gdy dzielimy na 2. Dlatego pesymistyczny czas to dalej jest O(n) i domyślam się że są takie przypadki w datasetcie od Huaweia.

Dzielenie na podprzestrzenie też może być słabe, bo są równe linie mogą się bardzo oddalić od punktów docelowych.
Trzeba na to zaradzić.

### Pomysł drugi - drzewa decyzyjne
Jak możemy zauważyć gdy robimy trik z binsearchem jesteśmy w stanie zmapować wszystkie punkty w postaci drzewa pytając się tylko do której podprzestrzeni wpisać dany punkt.
Czyli tworzymy pewnego rodzaju drzewo decyzyjne. Drzewa decyzyjne jak wiemy z MLa - są fajne, ale działają średnio, bo się łatwo przeuczają.
A jak udoskonalaliśmy te drzewa żeby były kox?

#### więcej drzew + import random = Random Forrest
Lasy losowe (ang. Random Forrest) pomagają nam w uczeniu drzew decyzyjnych, bo dzięki losowości jesteśmy w stanie unormować efekt przeuczenia drzew decyzyjnych.

#### jedno drzewo + boostowanko = XGBoost
Kolejnym pomysłem jest XGBoost - tak jak drzewa - pomaga nam ogarnąć drzewa decyzyjne więc może dawać nam lepsze wyniki niż standardowe drzewka.

### Pomysł trzeci - nie lubimy punktów żyjących w izolacji
Jak wiemy osoby żyjące w izolacji śmierdzą - tak samo jest z punktami.
Nie chcemy ich w naszym zbiorze, ale chcielibyśmy je zwrócić jeśli faktycznie są najbliżej (śmierdzące punkty trzymają się razem).

W znalezieniu tych śmierdzących punktów pomagają nam **Isolation Trees**. 

Jak już mamy je znalezione - możemy znaleźć odległości do nich i budować drzewo bez tych punktów.
Wtedy drzewo ma mniej nieregularnych gałęzi (jest pełniejsze(?)/bardziej zbalansowane) przez co skracamy czas wyszukiwania.

Nie czytałem jak sie je robi, więc możemy tak naprawdę użyć innych sposobów na szukanie śmierdzieli. ![wiki](https://en.wikipedia.org/wiki/Anomaly_detection#Methods)







