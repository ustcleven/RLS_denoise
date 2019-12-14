# Dual-Microphone Noise Removal for Keyword Spotting
### Frédéric Bischoff and Rayan Daod Nathoo
**Course**: _Audio and Acoustic Signal Processing_ by Christof Faller and Mihailo Kolundzija

**Deadline**: 23:59 on December 15th, 2019

## Getting Started

To run our project, please:
- Clone this project
- Download this data folder and place it at the root of the project:
- Run one of the two Jupyter Notebooks: Wiener or RLS

## Prerequisites
Python 3, Jupyter Notebook

## Inspiration

Since the past few years, speech recognition has become more and more present in our everyday lives. From the message you want to send while away from your cell phone to the analysis of phone calls for big companies, the speech processing field is expanding at a phenomenal speed. But let us focus on the personal use of automatic speech recognition (ASR). "Hey Siri", "Hey Google", and "Alexa" are the 3 most famous keywords to start a vocal query on their respective operating systems, and millions of data are collected every day on our cell phones to help the ASR algorithms to detect those keywords with high accuracy. However, Machine Learning is not the key to every problem and we still need signal processing to remove noise from the recorded signals to increase the F1-score. When talking about noise we mean environmental noise, background TV, radio, music, etc.

## What it does

We have implemented two Adaptive Noise Cancellation algorithms, based on Short-Time Fourier Transform (STFT):
- _STFT-domain Wiener filter with forgetting factor_
- _STFT-domain fast Recursive Least Squares (RLS) with deferred coefficients_

Basically, those two algorithms' goal is to remove the background noise from an audio signal but to let the speech signal go through, at least during the required amount of time for the following ASR algorithm to detect the keyword. Both are built under two assumptions: the short segment immediately preceding a keyword contains only noise, and a keyword has a short duration, typically less than 1 s.

## How we built it

For the Wiener filter, we based ourselves on scientific papers we found on the internet (c.f bibliography in the report) and on the explanations given by the teachers during the course. The second algorithm was mainly taken from two existing papers written by Google engineers (c.f bibliography in the report).

## Challenges we ran into

The most challenging part was to make the algorithms converge and to understand their behavior. Since it is the first time we are working on an audio signal processing project, we had a hard time finding the right libraries and figuring out why our implementations did not work at first. It required a lot of perseverance and discussions with the teachers to finally reach the end.

## Accomplishments that we are proud of

In the end, we implemented 2 (kinda) working signal processing algorithms almost without having got any help on the implementation part and being able to hear the result is always satisfying.

## What we learned

We learned the basics of signal processing algorithms implementation in Python, and get deeper with the understanding of adaptive filtering and audio in general.

## What's next for Dual-Microphone Noise Removal for Keyword Spotting
We did not have time to vectorize the algorithms which would make them a lot faster so this would be a possible future work. The logical continuation would then be to implement those algorithms in real-time and to combine them with actual keyword detection algorithms.
