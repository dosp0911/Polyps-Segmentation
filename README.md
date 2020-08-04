# Polyps-Segmentation
Pixel-wise image segmentation is a highly demanding task in medical image analysis. It is difficult to find annotated medical images with corresponding segmentation mask. Here, we present Kvasir-SEG. It is an open-access dataset of gastrointestinal polyp images and corresponding segmentation masks, manually annotated and verified by an experienced gastroenterologist. This work will be valuable for researchers to reproduce results and compare their methods in the future. By adding segmentation masks to the Kvasir dataset, which until today only consisted of framewise annotations, we enable multimedia and computer vision researchers to contribute in the field of polyp segmentation and automatic analysis of colonoscopy videos.

#### Kvasir-Segement Dataset
### Background
The human gastrointestinal (GI) tract is made up of different sections, one of them being the large bowel. Several types of anomalies and diseases can affect the large bowel, such as colorectal cancer. Colorectal cancer is the second most common cancer type among women and third most common among men. Polyps are precursors to colorectal cancer, and is found in nearly half of the individuals at age 50 having a screening colonoscopy, and are increasing with age. Colonoscopy is the gold standard for detection and assessment of these polyps with subsequent biopsy and removal of the polyps. Early disease detection has a huge impact on survival from colorectal cancer, and polyp detection is therefore important. In addition, several studies have shown that polyps are often overlooked during colonoscopies, with polyp miss rates of 14%-30% depending on the type and size of the polyps. Increasing the detection of polyps has been shown to decrease risk of colorectal cancer. Thus, automatic detection of more polyps at an early stage can play a crucial role in improving both prevention of and survival from colorectal cancer. This is the main motivation behind the development of a Kvasir-SEG dataset.
Original Kvasir Dataset Details
The Kvasir dataset comprises 8000 gastrointestinal (GI) tract images, each class consisting of 1000 images. These images were collected and verified by experienced gastroenterologist from Vestre Viken Health Trust in Norway. The eight classes of the dataset include anatomical landmarks, pathological findings and endoscopic procedures. Each class' images are saved in a separate folder corresponding to the class they belong to. More detailed explanation about each image class, data collection procedure and dataset details can be found on the Kvasir dataset's homepage at https://datasets.simula.no/kvasir/.

Some examples of the dataset.

<img src="https://datasets.simula.no/kvasir-seg/images/mix1.png" width="600px" align="center">
<img src="https://datasets.simula.no/kvasir-seg/images/mix2.png" width="600px" align="center">
<img src="https://datasets.simula.no/kvasir-seg/images/mix3.png" width="600px" align="center">

### Unet, ResUnet, ResUnet++ Model used
