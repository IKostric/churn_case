# Datasett – teknisk case

Syntetiske data for et utvalg personkunder. Snapshot-dato: **31.12.2025**.
Målvariabelen `avgang_6mnd` angir om kundeforholdet ble avsluttet i **januar–juni 2026**
(1 = avgang). Merk at målvariabelen gjelder samme fremtidsperiode for alle kunder –
vær bevisst på hva som faktisk ville vært kjent på snapshot-tidspunktet.

Dataene er syntetiske, men har realistiske datakvalitetsutfordringer. Håndteringen av
disse er en del av oppgaven.

## kunder.csv
Én rad per kunde (eksport fra kjernesystem).
- kunde_id, alder, fylke, kundeforhold_aar, antall_produkter
- antall_produkter_naa (antall produkter kunden har per i dag, uttrekksdato for filen)
- har_kundeprogram, laanebalanse (kr), rente_boliglaan (%), innskudd (kr)
- raadgiversamtaler_12mnd, avslutningsgebyr (kr), avgang_6mnd (mål)

## engasjement.csv
Månedlig digital aktivitet per kunde, jan. 2024–des. 2025.
- kunde_id, maaned (ÅÅÅÅ-MM), innlogginger, korttransaksjoner

## henvendelser.csv
Kundehenvendelser (fritekst) i perioden, med kanal og dato.
- henvendelse_id, kunde_id, dato, kanal, tekst

## relasjoner.csv
Kjente relasjoner mellom kunder (husstand / medlåntaker).
- kunde_id_1, kunde_id_2, relasjon
