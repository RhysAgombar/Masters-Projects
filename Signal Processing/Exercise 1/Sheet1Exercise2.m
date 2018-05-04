%% Instructions to Run
% Press the run button
%% A
F = 1000; % Frequency
Fs = 44100; % Frequency of Sampling
rate = 1/Fs; % time between samples
t = (0:rate:1)'; % timeline from 0 to 1 seconds

Data = sin(2*pi*F*t); % Generate sin wave

%% B
figure();
spectrogram(Data,'yaxis');
title("Sin Spectrogram");
%% C
DataCos = cos(2*pi*F*t);

figure();
spectrogram(DataCos,'yaxis');
title("Cos Spectrogram");
%% D
load handel.mat

file = 'D.wav';
audiowrite(file,Data,Fs);

file = 'Dc.wav';
audiowrite(file,DataCos,Fs);