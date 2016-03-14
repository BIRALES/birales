% ------------------------------------------------
% MULTI-BEAM BEAMFORMER 2-D
% BEST-2
% Autore: Giovanni Naldi <gnaldi@ira.inaf.it>
% IRA-INAF
% Ver: 27 Ottobre 2011
% ------------------------------------------------

% RESET ENVIRONMENT MATLAB
close all;      % Chiude tutte le FIGURE aperte...
clear all;      % Cancella tutte le VAR in memoria...
clc;            % Cancella lo schermo...

% CONFIGURAZIONE ARRAY (Si suppone di utilizzare sensori ideali)
N_x = 8;              % number of cylinders
d_x = 13.6;           % distance of cylinders
N_y = 4;              % number of receivers in 1 cylinder
d_y = 8;              % distance of receivers in 1 cylinder
N_dipoli = 16;
d_dipoli = 0.5;
offset = 0.000001;      % OFFSET in radianti contro angoli limite del dipolo.

min_dB = -80;


% LETTURA DA FILE DEL PATTERN D'ANTENNA SUL PIANO H DI UN CILINDRO
H_Plane_NS_Cyl = xlsread('NS_PianoH.xls');
Angle = H_Plane_NS_Cyl(:,2);
Angle = Angle(1761:1841);
%Angle = Angle(1:5:1801);
Pattern_dB = H_Plane_NS_Cyl(:,3);
Pattern_dB = Pattern_dB(1761:1841);
%Pattern_dB = Pattern_dB(1:5:1801);
HPlanePatternCyl = 10.^(Pattern_dB./20);
HPlanePatternCyl = HPlanePatternCyl./max(HPlanePatternCyl);
HPlanePatternCyl = abs(HPlanePatternCyl).^2;


% INIZIALIZZAZIONE DELLE VARIABILI SPAZIALI
% per il plottato 3D XYZ 
%offset = 0.000000001;
theta_x_min = -4;
theta_x_max = +4;
theta_x_step = 1;
MM=81;
theta_y_min = -4;
theta_y_max = +4;
theta_y_step = 1;
NN=81;

dtor=pi/180;                                          % fattore di conversione in radianti

theta_x_min_rad = theta_x_min*dtor;
theta_x_max_rad = theta_x_max*dtor;
theta_x_step_rad = theta_x_step/180*pi;
theta_y_min_rad = theta_y_min*dtor;
theta_y_max_rad = theta_y_max*dtor;
theta_y_step_rad = theta_y_step/180*pi;


% COSTRUZIONE DEL PIANO SPAZIALE
% per il plottato 3D XYZ
theta_x = linspace(theta_x_min,theta_x_max,MM);
theta_y = linspace(theta_y_min,theta_y_max,MM);
theta_x_rad = linspace(theta_x_min_rad,theta_x_max_rad,MM);
theta_y_rad = linspace(theta_y_min_rad,theta_y_max_rad,MM);
%dtheta=2*pi/MM;
%dphi=(pi/2)/NN;
[THETA_X_RAD,THETA_Y_RAD] = meshgrid(theta_x_rad,theta_y_rad);
[THETA_X,THETA_Y] = meshgrid(theta_x,theta_y);
%[PHI_RAD,THETA_RAD] = meshgrid(phi_rad,theta_rad);

%THETA = theta'*ones(1,N);


% ARRAY GEOMETRY MATRIX (ANTENNA PLACEMENTS)
X = zeros(N_x,N_y);
Y = zeros(N_x,N_y);
for m = 1:N_y
    for n = 1:N_x
        X(n,m) = (n-1)*d_x;
    end;
end;
for n = 1:N_x
    for m = 1:N_y
        Y(n,m) = (m-1)*d_y;
    end;
end;


N = N_x*N_y;                        % N = numero di antenne che compoingono la matrice

XY = zeros(N,2);
XY(:,1) = reshape(X,N,1);
XY(:,2) = reshape(Y,N,1);


% CALCOLO BEAMPATTERN DEL DIPOLO
beampattern_dipolo = cos((pi/2)*sin(theta_y_rad+offset))./cos(theta_y_rad+offset);
beampattern_dipolo = abs(beampattern_dipolo).^2;


% CALCOLO DEL FATTORE DI GRUPPO DEI DIPOLI
DIP = zeros(N_dipoli,length(theta_y_rad));
for a = 1:N_dipoli
    for b = 1:length(theta_y_rad)
    DIP(a,b) = exp(1i*2*pi*(a-1)*d_dipoli*sin(theta_y_rad(b)));
    end;
end;
w_dipoli = ones(N_dipoli,1)/N_dipoli;           % L'ACCOPPIAMENTO TRA I DIPOLI E' FISSO (PUNTAMENTO = 0°)
fattore_gruppo_dipoli = abs(w_dipoli'*DIP).^2;


% CALCOLO DEL BEAMPATTERN DEL SENSORE
beampattern_sensore = beampattern_dipolo.*fattore_gruppo_dipoli;

% CALCOLO DEL BEAM DEL SINGOLO ELEMENTO DEL BEST-2
beam_element = HPlanePatternCyl*beampattern_sensore;
beam_element = 10*log10(beam_element);
beampattern_sensore = beampattern_sensore.';

AF3D = zeros(N,length(theta_x_rad),length(theta_y_rad));
for b = 1:length(theta_y_rad) 
    for a = 1:length(theta_x_rad)
        for n = 1:N
            AF3D(n,a,b) = exp(1i*2*pi*((XY(n,1)*sin(theta_x_rad(a)))+(XY(n,2)*sin(theta_y_rad(b)))));
        end;
        %temp_field=temp_field.*felem;
        %AF3D(a,b)=abs(temp_field);
        %AF3D(a,b)=temp_field;
    end;
end;


AF3D_mod = reshape(AF3D,N_x,N_y,length(theta_x_rad),length(theta_y_rad));



fft_AF3D_mod = zeros(N_x,N_y,length(theta_x_rad),length(theta_y_rad));
fft_temp = zeros(N_x,N_y);
for b = 1:length(theta_y_rad) 
    for a = 1:length(theta_x_rad)
        fft_temp = squeeze(AF3D_mod(:,:,a,b));
        fft_temp = fft2(fft_temp);
        fft_AF3D_mod(:,:,a,b) = fft_temp;
    end;
end;

fft_AF3D_mod = abs(fft_AF3D_mod).^2;
fft_AF3D_mod_max = max(max(max(max(fft_AF3D_mod))));
fft_AF3D_mod = fft_AF3D_mod./fft_AF3D_mod_max;

clear fft_temp;
for a = 1:N_x
    for b = 1:N_y
        for c = 1:length(theta_x_rad)
            fft_temp = squeeze(fft_AF3D_mod(a,b,c,:));
            fft_temp = fft_temp.*beampattern_sensore;
            fft_AF3D_mod(a,b,c,:) = fft_temp;
        end;
    end;
end;

clear fft_temp;
for a = 1:N_x
    for b = 1:N_y
        for c = 1:length(theta_y_rad)
            fft_temp = squeeze(fft_AF3D_mod(a,b,:,c));
            fft_temp = fft_temp.*HPlanePatternCyl;
            fft_AF3D_mod(a,b,:,c) = fft_temp;
        end;
    end;
end;

fft_AF3D_mod = 10*log10(fft_AF3D_mod);

for a = 1:N_x
    for b = 1:N_y
        for c = 1:length(theta_x_rad)
            for d = 1:length(theta_y_rad)
                if (fft_AF3D_mod(a,b,c,d)<min_dB)
                    fft_AF3D_mod(a,b,c,d) = min_dB;
                end;
            end;
        end;
    end;
end;


%{
% PLOT
clear fft_temp;
fft_temp = zeros(length(theta_x_rad),length(theta_y_rad));
for b = 1:N_y
    for a = 1:N_x
        fft_temp = squeeze(fft_AF3D_mod(a,b,:,:));
        clims = [ -80 0 ];
        surfc(THETA_Y,THETA_X,fft_temp);
        view(45,45);
        colorbar;
        hold on;
    end;
end;
grid on;
title('FFT BEAMFORMER OF 2D ARRAY');
xlabel('Declination (H-Plane) [°]');
ylabel('Right Ascension (E-Plane) [°]');
zlabel('G [dB]');
%axis([-100 100 -100 100 -80 0]);



figure;
clear fft_temp;
fft_temp = zeros(length(theta_x_rad),length(theta_y_rad));
for b = 1:N_y
    for a = 1:N_x
        fft_temp = squeeze(fft_AF3D_mod(a,b,:,:));
        clims = [ -80 0 ];
        surfc(THETA_Y,THETA_X,fft_temp);
        view(0,0);
        colorbar;
        hold on;
    end;
end;
grid on;
title('FFT BEAMFORMER OF 2D ARRAY');
xlabel('Declination (H-Plane) [°]');
ylabel('Right Ascension (E-Plane) [°]');
zlabel('G [dB]');
%axis([-100 100 -100 100 -80 0]);

figure;
clear fft_temp;
fft_temp = zeros(length(theta_x_rad),length(theta_y_rad));
for b = 1:N_y
    for a = 1:N_x
        fft_temp = squeeze(fft_AF3D_mod(a,b,:,:));
        clims = [ -80 0 ];
        surfc(THETA_Y,THETA_X,fft_temp);
        view(90,0);
        colorbar;
        hold on;
    end;
end;
grid on;
title('FFT BEAMFORMER OF 2D ARRAY');
xlabel('Declination (H-Plane) [°]');
ylabel('Right Ascension (E-Plane) [°]');
zlabel('G [dB]');
%axis([-100 100 -100 100 -80 0]);

figure;
clear fft_temp;
fft_temp = zeros(length(theta_x_rad),length(theta_y_rad));
for b = 1:N_y
    for a = 1:N_x
        fft_temp = squeeze(fft_AF3D_mod(a,b,:,:));
        clims = [ -80 0 ];
        surfc(THETA_Y,THETA_X,fft_temp);
        view(0,90);
        colorbar;
        hold on;
    end;
end;
grid on;
title('FFT BEAMFORMER OF 2D ARRAY');
xlabel('Declination (H-Plane) [°]');
ylabel('Right Ascension (E-Plane) [°]');
zlabel('G [dB]');
%axis([-100 100 -100 100 -80 0]);

figure;
clear fft_temp;
fft_temp = zeros(length(theta_x_rad),length(theta_y_rad));
for b = 1:N_y
    for a = 1:N_x
        fft_temp = squeeze(fft_AF3D_mod(a,b,:,:));
        clims = [ -80 0 ];
        contour(THETA_Y,THETA_X,fft_temp);
        colorbar;
        hold on;
    end;
end;
grid on;
title('FFT BEAMFORMER OF 2D ARRAY');
xlabel('Declination (H-Plane) [°]');
ylabel('Right Ascension (E-Plane) [°]');
%}


%figure;
clear fft_temp;
fft_temp = zeros(length(theta_x_rad),length(theta_y_rad));
beam_number = 1;
title_beam = 'BEAM NUMBER = ';
label_beam = 'beam ';

h_prec = [];
%for b = 1:1
for b = 1:N_y
    %for a = 1:1
    for a = 1:N_x
        fft_temp = squeeze(fft_AF3D_mod(a,b,:,:));
        [C,h] = contour(THETA_Y,THETA_X,fft_temp,[-3 -3]);
        
        
        clabel(C,h,'FontSize',14,'Color','r','Rotation',0,'FontWeight','bold');
        h = findobj('Type','text');
        etichetta_beam = [label_beam, num2str(beam_number)];
        if (length(h)-length(h_prec)==1)
            set(h(1),'String',num2str(beam_number));
        else
            set(h(1:(length(h)-length(h_prec))),'String',num2str(beam_number));
        end;
        
        
        colormap winter;
        %colorbar;
        hold on;
        
        title([title_beam, num2str(beam_number)]);
        %pause;
        beam_number = beam_number+1;
        clear h_prec
        h_prec = h;
    end;
end;
grid on;
title('FFT BEAMFORMER OF BEST-2 ARRAY');
xlabel('Declination (H-Plane) [°]');
ylabel('Right Ascension (E-Plane) [°]');

hold on;
%figure;
contour(THETA_Y,THETA_X,beam_element,[-3 -3]);
colorbar;
grid on;

