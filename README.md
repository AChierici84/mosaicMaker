# Mosaic Maker
Un lego mosaic maker per le immagini basato sul sito: [https://www.lego.com/it-it/mosaic-maker](https://www.lego.com/it-it/mosaic-maker)

L'idea è quella di trasformare un'immagine in un mosaico lego 48x48 con 5 colori base:
* bianco
* giallo
* nero
* grigio chiaro
* grigio scuro

Il processo è stato replicato con l'utilizzo di un cluster k-means e delle liberie open-cv.

Data un'immagine in input vengono aggiustate le dimensioni . L'immagine tagliata per avere una dimensione 1 : 1.
Quindi si procede al clustering con i 5 colori base, assegnati randomicamente al posto dei cluister center. 
Infine l'immagine viene divisa in 48 x 48 blocchi e per ognuno viene selezionato il colore col numero maggiore di occarenze nel blocco.
La griglia risultante 48X48 rende possibile il conteggio dei pezzi necessari per realizzare l'immagine su una base lego
della stessa dimensione e visualizza al loro disposizione in una griglia per facilitare il montaggio.

Viene quindi salvato un file .jpg con la griglia istruzioni e un file .txt coi conteggi dei mattoncini necezzari per ogni colore. 

Di seguito alcuni esempi di risultati.

|  Immagine orginale | Mosaico               |
|--------------------|-----------------------|
|![gatto lego](https://github.com/user-attachments/assets/2df0e4b6-2676-40a7-8664-50efbab39dc6)|<img width="1440" height="1440" alt="final_lego_mosaic" src="https://github.com/user-attachments/assets/63474fc1-5d96-4abe-a977-5dab1265596c" />|
|![rosa sboccia](https://github.com/user-attachments/assets/8f3a1945-8877-4a45-9ebb-b79e739168d6)|<img width="1440" height="1440" alt="rosa sboccia_mosaic" src="https://github.com/user-attachments/assets/024c33de-6f61-40bb-809a-a0a6a2e4361d" />|
|![migliori-marche-auto-in-europa-e-nel-mondo_3](https://github.com/user-attachments/assets/3e4bb460-4d4d-41c1-891b-942f935bc26e)|<img width="1440" height="1440" alt="migliori-marche-auto-in-europa-e-nel-mondo_3_mosaic" src="https://github.com/user-attachments/assets/6d9277fc-2086-4558-959e-15ff66731508" />|
|![154942029-0ae430e6-bb13-49ac-9e14-100961fb04fb](https://github.com/user-attachments/assets/750088fb-9728-4f3b-8805-cf33bc93fd9f)|<img width="1440" height="1440" alt="auto_mosaic" src="https://github.com/user-attachments/assets/169d94e4-abd1-4c3e-b56d-ed46ed899d82" />|
|![For-Those-About-To-Rock2](https://github.com/user-attachments/assets/e923e532-af3f-4c10-9c28-24cba472d6be)|<img width="1440" height="1440" alt="For-Those-About-To-Rock2_mosaic" src="https://github.com/user-attachments/assets/9a1880a0-bb44-4fd9-a477-cb3260ec90ab" />|


Seguendo le griglie riportate è possibile realizzare qualaisasi immagine anche foto o ritratti comprando l'apposito kit sul sito della lego.
Rispetto alla versione originale il taglio delle foto è autoamico alla dimensione inferiore e viene tenuta sempre la parte cnetrale dell'immagine non è possibile selezionare un'area differente.

Essendo presente un'assegnazione randomica dei cluster ai 5 colori base si può provare a generare diverse combinazioni e scegliere poi la migliore. 
Il seed dei k-mean è stato fissato al valore 42 ma variandolo si possono generare anche maggiori combinazioni. 

L'applicazione realizzata è un modulo python che può essere inglobato in interfacce web o altri contesti. 

Punti di miglioramenta:
 * pulizia del background
 * in caso di più soggetti rendere possibile selezionare il soggetto desiderato

**Nota**: per scegliere la griglia da realizzare si consiglia sempre di osservarla da una certa distanza perchè il mosaico è pensato per essere appeso a una parete.









