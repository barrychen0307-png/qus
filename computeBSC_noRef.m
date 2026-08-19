function results = computeBSC_noRef(rf_roi, fs, varargin)
% computeBSC_noRef  Reference-free backscatter spectral analysis on an ROI
%                   with depth-dependent attenuation compensation. (v2)
%
%   results = computeBSC_noRef(rf_roi, fs)
%   results = computeBSC_noRef(rf_roi, fs, 'Name', Value, ...)
%
% Inputs
% ------
%   rf_roi : [Nz x Nx] RF data inside a rectangular ROI
%            Nz = axial samples, Nx = number of RF lines (scan lines)
%   fs     : sampling frequency in Hz
%
% Name-Value pairs
% ----------------
%   'window'     : 'hann' (default) | 'hamming' | 'rect'
%   'bandwidth'  : [fmin fmax] in Hz. Default = automatic -6 dB band.
%   'subWinPct'  : axial sub-window length as fraction of Nz (default 1.0).
%                  If <1, use overlapping sub-windows (Welch-style).
%   'overlap'    : sub-window overlap, 0..0.95 (default 0.5)
%   'plot'       : true | false (default false)
%
%   --- v2: attenuation compensation ---
%   'attenComp'  : true (default) | false   - enable / disable comp.
%   'alpha0'     : attenuation coefficient (dB/cm/MHz, one-way pressure).
%                  Default 0.5 (typical soft tissue).
%                  Frequency model: alpha(f) = alpha0 * f_MHz (linear).
%   'soundSpeed' : speed of sound in m/s. Default 1540.
%   'startDepth' : ROI top depth in METERS (round-trip distance / 2).
%                  Required when attenComp is true; if omitted defaults to
%                  0 with a warning.
%
% Outputs (struct)
% ----------------
%   freq         : frequency axis (Hz, single-sided)
%   PS           : averaged power spectrum (linear, attenuation-compensated
%                  if attenComp = true)
%   PS_dB        : PS in dB
%   PS_raw       : averaged power spectrum WITHOUT compensation (for ref.)
%   PS_raw_dB    : PS_raw in dB
%   f_band       : frequencies inside the analysis band
%   PS_band_dB   : compensated spectrum inside the analysis band (dB)
%   slope        : spectral slope (dB/MHz)         [Lizzi-Feleppa]
%   intercept    : spectral intercept (dB)         [extrapolated to f=0]
%   midband      : mid-band fit (dB)               [fit value at band center]
%   BSC_rel      : relative BSC = PS / max(PS) inside band (unitless, 0..1)
%   bandwidth    : [fmin fmax] used for fitting (Hz)
%   atten        : substruct with attenuation settings actually used
%
% NOTES on what "reference-free" means here
% -----------------------------------------
% Without a reference phantom (or planar reflector) the ABSOLUTE BSC
% in units of (1/cm.sr) cannot be recovered, because the system terms
%
%      W_meas(f) = G(f) * |H(f)|^2 * D(f) * A(f) * BSC(f)
%
% remain unknown.  This routine returns:
%   (1) the calibrated power-spectrum estimate of the ROI (after removing
%       the depth-dependent attenuation A(f) when attenComp = true),
%   (2) Lizzi-Feleppa spectral parameters (slope/intercept/midband),
%       which are approximately system-independent when the system
%       response is roughly flat over the analysis band, and
%   (3) a self-normalized relative BSC (PS / max(PS)) that preserves
%       the frequency dependence of the scattering.
%
% Attenuation compensation (v2)
% -----------------------------
% For each Welch sub-window centred at depth z_seg (m), the periodogram
% is multiplied by the round-trip POWER compensation factor
%
%     comp_lin(f, z_seg) = 10^( 0.2 * alpha0 * f_MHz * z_seg_cm )
%     <=>  comp_dB (f, z_seg) = 2 * alpha0 * f_MHz * z_seg_cm
%
% before being averaged.  The factor 2 (instead of 4) collects the
% round-trip and the amplitude<->power dB equivalence; see the user
% guide for the derivation.
%
% Author: example implementation, v2 (adds depth-dependent attenuation
% compensation, startDepth / alpha0 / soundSpeed parameters)
% -------------------------------------------------------------------------

% ---------- parse inputs ----------
p = inputParser;
addRequired(p,  'rf_roi', @(x) isnumeric(x) && ismatrix(x));
addRequired(p,  'fs',     @(x) isnumeric(x) && isscalar(x) && x>0);
addParameter(p, 'window',     'hann', @(s) any(strcmpi(s,{'hann','hamming','rect'})));
addParameter(p, 'bandwidth',  [],     @(x) isempty(x) || (numel(x)==2 && x(2)>x(1)));
addParameter(p, 'subWinPct',  1.0,    @(x) isscalar(x) && x>0 && x<=1);
addParameter(p, 'overlap',    0.5,    @(x) isscalar(x) && x>=0 && x<0.95);
addParameter(p, 'plot',       false,  @islogical);
% v2 attenuation parameters
addParameter(p, 'attenComp',  true,   @islogical);
addParameter(p, 'alpha0',     0.5,    @(x) isscalar(x) && x>=0);
addParameter(p, 'soundSpeed', 1540,   @(x) isscalar(x) && x>0);
addParameter(p, 'startDepth', [],     @(x) isempty(x) || (isscalar(x) && x>=0));
parse(p, rf_roi, fs, varargin{:});
opt = p.Results;

[Nz, Nx] = size(rf_roi);
if Nz < 16
    error('Axial dimension is too short (Nz=%d) for spectral analysis.', Nz);
end

% startDepth handling
if opt.attenComp
    if isempty(opt.startDepth)
        warning(['attenComp is true but startDepth was not provided. ' ...
                 'Assuming startDepth = 0 m (transducer face). ' ...
                 'Pass ''startDepth'', z_top to remove this warning.']);
        z0 = 0;
    else
        z0 = opt.startDepth;
    end
else
    z0 = 0;   % unused
end

% remove DC per line
rf_roi = rf_roi - mean(rf_roi, 1);

% sub-window length (Welch-style averaging in the axial direction)
Lwin = max(16, round(opt.subWinPct * Nz));
hop  = max(1, round(Lwin * (1 - opt.overlap)));
seg_starts = 1 : hop : (Nz - Lwin + 1);
if isempty(seg_starts), seg_starts = 1; Lwin = Nz; end

% window
switch lower(opt.window)
    case 'hann',    w = hann(Lwin);
    case 'hamming', w = hamming(Lwin);
    case 'rect',    w = ones(Lwin,1);
end
U = sum(w.^2) / Lwin;        % window-energy normalisation factor

% FFT length
Nfft = 2^nextpow2(max(256, Lwin));
PS_acc     = zeros(Nfft,1);
PS_acc_raw = zeros(Nfft,1);   % uncompensated, for diagnostics
nseg       = 0;

% pre-compute frequency vector (full FFT bins, in MHz, for compensation)
f_full_Hz  = (0:Nfft-1).' * (fs / Nfft);
f_full_MHz = f_full_Hz / 1e6;

% ---------- main loop: accumulate periodograms ----------
for ix = 1:Nx
    line = rf_roi(:,ix);
    for s = seg_starts
        seg = line(s : s+Lwin-1) .* w;
        F   = fft(seg, Nfft);
        Pseg = abs(F).^2;

        if opt.attenComp
            % depth at the centre of this sub-window (round-trip / 2)
            % sample index of segment centre (1-based):
            sc       = s + (Lwin-1)/2;
            t_center = (sc - 1) / fs;                          % seconds
            z_seg    = z0 + 0.5 * opt.soundSpeed * t_center;   % metres
            z_seg_cm = z_seg * 100;
            % round-trip power compensation in linear scale
            comp_lin = 10.^(0.2 * opt.alpha0 .* f_full_MHz .* z_seg_cm);
            PS_acc   = PS_acc   + Pseg .* comp_lin;
        else
            PS_acc   = PS_acc   + Pseg;
        end
        PS_acc_raw   = PS_acc_raw   + Pseg;
        nseg = nseg + 1;
    end
end
PS_full     = PS_acc     / (nseg * Lwin * U);    % unbiased PSD (a.u.)
PS_full_raw = PS_acc_raw / (nseg * Lwin * U);

% single-sided spectrum
half = 1 : Nfft/2;
freq = (half-1).' * (fs / Nfft);
PS       = 2*PS_full(half);     PS(1)     = PS(1)/2;
PS_raw   = 2*PS_full_raw(half); PS_raw(1) = PS_raw(1)/2;
PS_dB     = 10*log10(PS     + eps);
PS_raw_dB = 10*log10(PS_raw + eps);

% ---------- analysis band ----------
if isempty(opt.bandwidth)
    [pk, ipk] = max(PS_dB);
    % connected band around the peak (-6 dB)
    lo = ipk;  while lo>1            && PS_dB(lo-1) >= pk-6, lo = lo-1; end
    hi = ipk;  while hi<numel(PS_dB) && PS_dB(hi+1) >= pk-6, hi = hi+1; end
    f_lo = freq(lo);  f_hi = freq(hi);
else
    f_lo = opt.bandwidth(1);  f_hi = opt.bandwidth(2);
end
band = freq >= f_lo & freq <= f_hi;
if nnz(band) < 4
    warning('Analysis band has too few bins (%d). Widen the bandwidth.', nnz(band));
end
f_band   = freq(band);
PSb_dB   = PS_dB(band);

% ---------- Lizzi-Feleppa parameters (linear fit in dB) ----------
fMHz   = f_band / 1e6;
coef   = polyfit(fMHz, PSb_dB, 1);   % coef(1) = slope (dB/MHz)
slope     = coef(1);
intercept = coef(2);
fc_band   = mean([f_lo f_hi]) / 1e6;
midband   = polyval(coef, fc_band);

% ---------- relative (self-normalised) BSC ----------
PS_band_lin = PS(band);
BSC_rel = PS_band_lin / max(PS_band_lin);

% ---------- pack ----------
results = struct();
results.freq        = freq;
results.PS          = PS;
results.PS_dB       = PS_dB;
results.PS_raw      = PS_raw;
results.PS_raw_dB   = PS_raw_dB;
results.f_band      = f_band;
results.PS_band_dB  = PSb_dB;
results.slope       = slope;          % dB/MHz
results.intercept   = intercept;      % dB
results.midband     = midband;        % dB at band centre
results.BSC_rel     = BSC_rel;        % unitless (0..1) inside band
results.bandwidth   = [f_lo f_hi];
results.nSegments   = nseg;
results.nLines      = Nx;
results.atten       = struct( ...
    'enabled',    opt.attenComp, ...
    'alpha0',     opt.alpha0, ...
    'unit',       'dB/cm/MHz (one-way)', ...
    'soundSpeed', opt.soundSpeed, ...
    'startDepth', z0);

% ---------- optional plot ----------
if opt.plot
    figure('Color','w','Name','Reference-free BSC analysis (v2)');
    subplot(1,2,1);
    plot(freq/1e6, PS_raw_dB, 'Color',[.7 .7 .7]); hold on;
    plot(freq/1e6, PS_dB,    'Color',[.3 .3 .3]);
    plot(f_band/1e6, PSb_dB, 'b', 'LineWidth',1.5);
    plot(fMHz, polyval(coef,fMHz), 'r--', 'LineWidth',1.5);
    xline(f_lo/1e6,':k'); xline(f_hi/1e6,':k');
    xlabel('Frequency (MHz)'); ylabel('Power (dB)');
    if opt.attenComp
        ttl = sprintf('Comp. ON (\\alpha_0=%.2f dB/cm/MHz)\nSlope=%.2f dB/MHz, Mid=%.2f dB', ...
            opt.alpha0, slope, midband);
    else
        ttl = sprintf('Comp. OFF\nSlope=%.2f dB/MHz, Mid=%.2f dB', slope, midband);
    end
    title(ttl);
    legend({'raw','compensated','fit band','LF fit'},'Location','best');
    grid on;

    subplot(1,2,2);
    plot(f_band/1e6, BSC_rel, 'b', 'LineWidth',1.5);
    xlabel('Frequency (MHz)'); ylabel('BSC_{rel} (normalised)');
    title('Self-normalised relative BSC');
    grid on;
end
end
