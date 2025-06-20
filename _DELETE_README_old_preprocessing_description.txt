# ROCKET TSC

## data preprocessing

### pp_1_elba
- einheitliches Umbenennen der columns und labels
- definieren des Timestamps als DatetimeIndex
- drop Cable_E2 (fehlerhafter Versuch)
- binary, ternary und anomaly label werden von scenario_label column abgeleitet und hinzugefügt

### pp_2_elba
- ersetz einen DatetimeIndex, der fehlerhaftes Format hat
- alle Sensorknoten erhalten den selben Startzeitpunkt (min_start_time)
- es wird pro Sensorknoten geresampled auf eine exakte sampling rate von 1/10s (mittels ffill)
- Wegschneiden des Endes der Zeitreihe, da Sensorknoten zeitversetzt ausgeschaltet wurden, sodass jede Zeitreihe pro Sensorknoten gleiche Länge hat
- justieren der Labels: Original Labels wurden teilweise zu früh gesetzt, justierung mittels Grenzwerten von PM05 und PM10 (das wird über 'fire_label_control' gesetzt, sodass die original Labels erhalten bleiben)
- drop Motion_Room

### pp_3_elba
- anpassen aller Label column auf Grundlage des 'fire_label_control' (progess_label und experiment_number werden original belassen)
- (Umsortieren der Reihenfolge bzw. kopieren der Fire und Nuisance Events (so, dass im Training alle Ereignisse mind. 2 mal und im Test mind. 1 mal vertreten sind), so dass in Abh. Intervall-Länge folgendes möglich ist:
	- Ein Datensatz wo nur ein Feuer drin (oder keins) in einem Intervall
	- Ein Datensatz  (wo eins oder mehrere Feuer Events in einem Intervall))

## pp_4_elba
- drop der Lüftungsphase nach jedem Event
- erzeugen Intervalle für ROCKET (definieren der Intervall-Länge + ob überlappende Intervalle)
- (zusätzliche Features, z.B. Trend feature ableiten)
- Export als ROCKET compatible format
- train/test split in time (export train and test df separately)