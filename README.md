## Automatic Score Viewer and Subtitle Display in Real-time Opera Tracking

This repository contains the corresponding code for our extended abstract:

>[Brazier C.](https://www.jku.at/en/institute-of-computational-perception/about-us/people/charles-brazier/)  and 
>[Widmer G.](https://www.jku.at/en/institute-of-computational-perception/about-us/people/gerhard-widmer/) <br>
"[Automatic Score Viewer and Subtitle Display in Real-time Opera Tracking]()".<br>
*Proposed as Extended Abstracts for the Late-Breaking Demo Session of the 23nd International Society for Music Information Retrieval Conference*, 2022


### Videos

The folder [`videos`](https://github.com/chorlybwaz/opera_tracker/tree/master/videos) contains several recordings of our real-time opera tracker during the playback of different YouTube videos in different scenarios. It includes one excerpt with a transition containing applause and a skipped part, and two trackings of isolated parts tracked with two different features. For a complete opera tracking, please look at our [dropbox link](https://www.dropbox.com/s/12gjj221qowvrfc/Act1_CompleteTracking.mp4).



## Getting Started

The code is built with PyQt5. To play around with it, follow the instructions below.


### Installation

Clone the repository:
`git clone https://github.com/chorlybwaz/opera_tracker.git`

Move to the cloned folder:
`cd opera_tracker`

Intall the anaconda environment:
`conda env create -f environment.yml`

Activate the environment:
`conda activate opera_tracker`


### Real-time Tracking

This repository contains several runnable applications:
- `OperaTracker_Full.py` is designed to track complete operas. It handles structural mismatches and spontaneous applause. It is also possible to try out three different alignment algorithms (OLTW, JOLTW, JOLTWLR).
- `OperaTracker_Part.py` is designed to track isolated parts of the opera with either Posteriogram or MFCC features.
- `applause_detector.py` showcases with a green/red button the output of our applause detector.


## Acknowledgements

Special thanks to Christopher Widauer for providing the Don Giovanni lyrics annotations from the Vienna State Opera.

The research is supported by the European Union under the EU's Horizon 2020 research and innovation programme, Marie Skłodowska-Curie grant agreement No.765068 ([MIP-Frontiers](https://mip-frontiers.eu/)). The LIT AI Lab is supported by the Federal State of Upper Austria.

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Flag_of_Europe.svg/2560px-Flag_of_Europe.svg.png" width="20%" height="20%">
<img src="https://pbs.twimg.com/profile_images/1034814333899943936/0AnkUTWD_400x400.jpg" width="15%" height="15%">