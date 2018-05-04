%% Set up the signal
F = 10; % Frequency
Fs = 1024 - 1; % Frequency of Sampling
rate = 1/Fs; % time between samples
t = (0:rate:1)'; % timeline from 0 to 1 seconds
i = sqrt(-1);

Data = sin(2*pi*F*t) + i*cos(t); % Our signal

figure();

x = real(Data);
y = imag(Data);
plot(x, y, 'bo-', 'LineWidth', 2);
title('Plot of sin(2*pi*F*t) + i*cos(t)'); 
xlabel('Real Component');
ylabel('Imaginary Component');
grid on;

%% Visualizing the FT Functions

out = ckFFT(Data);

y2 = abs(out);
x2 = 0:1:length(y2)-1;
figure();
plot(x2, y2, 'bo-', 'LineWidth', 2);
title('Cooley Tukey Algorithm');
grid on;

test = fft(Data);
figure();
y2 = abs(test);
x2 = 0:1:length(y2)-1;
plot(x2, y2, 'bo-', 'LineWidth', 2);
title('Built in FFT Commmand');
grid on;

%% FFT Function
function out = ckFFT(data)

N = length(data); % This approach only works when the length of the data is a power of 2.

if N > 2
    x0 = data(2:2:end); % split into even and odd parts
    x1 = data(1:2:end);

    even = ckFFT(x0); % Recursion
    odd = ckFFT(x1);

    arr = (0:N/2-1)'; 
    dM = exp(-1i * 2 * pi * arr / N)

    uX = odd + dM .* even; % Process the upper and lower halves of 'x'
    lX = odd - dM .* even;

    out = vertcat(uX,lX); % Combine after processing and return
else 
    odd = data(1); % Same as above, just without recursion.
    even = data(2);
    
    arr = (0:N/2-1)'; 
    dM = exp(-1i * 2 * pi * arr / N)

    uX = odd + dM .* even;
    lX = odd - dM .* even;

    out = vertcat(uX,lX);
end

end