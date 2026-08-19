function main_qus_mega_batch_unbounded_fusion()
% ========================================================================
% 自適應多參數量化超音波 - 無限制中位數對決版 (Unbounded Median Release)
% 🚀 終極異質性升級版：全幀聚合 + 擾動特徵 + RFM + 論文級三聯圖匯出
% ========================================================================
clc; clear; close all; warning off;

%% 1. 系統硬體與物理參數設定
MHz = 10^6; cm = 10^(-2);
p.centerfreq = 3.5 * MHz;    
p.samplingrate = 12 * MHz;   
p.soundspeed = 1540;         
p.R0 = 6 * cm;               
p.probe_angle_deg = 60;      
p.capsule_abs_bsc = -20;     
p.compressionFactor = 60;    

%% 2. 選擇「大根目錄」
root_dir = uigetdir(pwd, '請選擇包含四個分級（如D1~D4）的「最外層大根目錄」');
if root_dir == 0, disp('❌ 已取消選擇'); return; end

all_items = dir(root_dir);
grade_subdirs = all_items([all_items.isdir] & ~strcmp({all_items.name}, '.') & ~strcmp({all_items.name}, '..'));
num_grades = length(grade_subdirs);

if num_grades == 0, error('❌ 找不到子資料夾，請確認目錄結構！'); end

fprintf('\n📂 頂層大目錄讀取成功！啟動【全幀異質性特徵 + 三聯圖】引擎。\n');

%% 3. 開始遍歷每一個分級資料夾
for g = 1:num_grades
    grade_name = grade_subdirs(g).name;
    grade_path = fullfile(grade_subdirs(g).folder, grade_name);
    
    fprintf('\n🎬 [分級任務 %d/%d] 開始解析 -> %s\n', g, num_grades, grade_name);
    
    patient_dirs = dir(grade_path);
    patient_dirs = patient_dirs([patient_dirs.isdir] & ~strcmp({patient_dirs.name}, '.') & ~strcmp({patient_dirs.name}, '..'));
    num_patients = length(patient_dirs);
    
    if num_patients == 0, continue; end
    
    output_img_dir = fullfile(pwd, ['ROI_Verification_Unbounded_', grade_name]);
    if ~exist(output_img_dir, 'dir'), mkdir(output_img_dir); end
    
    results_table = table(); 
    
    % 🌟 進入【病患層級】
    for p_idx = 1:num_patients
        patient_name = patient_dirs(p_idx).name;
        patient_path = fullfile(patient_dirs(p_idx).folder, patient_name);
        
        mat_files = dir(fullfile(patient_path, '*.mat'));
        num_frames = length(mat_files);
        
        if num_frames == 0, continue; end
        
        fprintf('    [%03d/%03d] 病患 %s (共 %d 張影像) 萃取中... ', p_idx, num_patients, patient_name, num_frames);
        
        % 第 6 欄用來儲存 SNR
        temp_features = zeros(num_frames, 6); 
        valid_frames = 0;
        
        % 🌟 進入【影像層級】
        for f_idx = 1:num_frames
            file_path = fullfile(mat_files(f_idx).folder, mat_files(f_idx).name);
            clean_save_name = sprintf('%s_frame%d', patient_name, f_idx);
            
            try
                data = load(file_path); vars = fieldnames(data);
                rf_raw = double(data.(vars{1}));
                if ndims(rf_raw) == 3, rf_raw = mean(rf_raw, 3); end 
                [n_samples, n_lines] = size(rf_raw);
                angle_span = deg2rad(p.probe_angle_deg);
                
                depth_axis = p.soundspeed * 0.5 * ((1:n_samples) / p.samplingrate);
                theta = linspace(-angle_span/2, angle_span/2, n_lines);
                R_physical = (depth_axis + p.R0) * 100; 
                [THETA, R_MAT] = meshgrid(theta, R_physical);
                X_sector = R_MAT .* sin(THETA); Z_sector = R_MAT .* cos(THETA) - p.R0*100;
                
                rf_env = abs(hilbert(rf_raw));
                img_log = 20 * log10(rf_env + eps);
                img_rect = img_log - max(img_log(:)) + p.compressionFactor;
                img_rect(img_rect < 0) = 0;
                
                z_cm = depth_axis * 100;
                search_mask = (z_cm >= 0.5) & (z_cm <= 2.5); 
                search_idx = find(search_mask);
                center_lines = round(n_lines/2) - 5 : round(n_lines/2) + 5;
                env_profile = mean(img_rect(:, center_lines), 2);
                [~, max_loc] = max(env_profile(search_mask));
                auto_capsule_depth = z_cm(search_idx(max_loc));
                
                roi_w = 2.5; roi_h = 1.2; x_start = -roi_w / 2;
                min_safe_z = depth_axis(1) * 100; 
                expected_r_top = auto_capsule_depth - (roi_h * 0.85); 
                current_roi_r = [x_start, max(min_safe_z, expected_r_top), roi_w, roi_h];
                
                [current_roi_t, target_snr] = auto_find_clean_target_roi(rf_raw, n_samples, n_lines, p, angle_span);
                
                % ====================================================
                % 1. 先執行所有核心物理運算 (不先畫圖)
                % ====================================================
                [block_T, z_top_m_T] = extract_rf_block(current_roi_t, rf_raw, n_samples, n_lines, p, angle_span);
                [block_R, z_top_m_R] = extract_rf_block(current_roi_r, rf_raw, n_samples, n_lines, p, angle_span);
                ac_roi = [current_roi_t(1), 1.5, current_roi_t(3), 4.0];
                [block_AC, ~] = extract_rf_block(ac_roi, rf_raw, n_samples, n_lines, p, angle_span);
                
                % 🎯 使用 RFM + 防呆機制計算衰減
                [alpha_est, alpha_raw] = estimateAC_RFM(block_AC, p.samplingrate, p.soundspeed, [2, 5]);
                
                % BSC 頻譜計算 (引入 RFM 算出的 alpha_est)
                bsc_t = computeBSC_noRef(block_T, p.samplingrate, 'window', 'hann', 'subWinPct', 0.5, 'overlap', 0.5, 'attenComp', true, 'alpha0', alpha_est, 'soundSpeed', p.soundspeed, 'startDepth', z_top_m_T, 'plot', false);
                bsc_r = computeBSC_noRef(block_R, p.samplingrate, 'window', 'hann', 'subWinPct', 0.5, 'overlap', 0.5, 'attenComp', true, 'alpha0', 0.5, 'soundSpeed', p.soundspeed, 'startDepth', z_top_m_R, 'plot', false);
                    
                f_axis = bsc_t.freq / MHz;
                final_bsc = (bsc_t.PS_dB - bsc_r.PS_dB) + p.capsule_abs_bsc;
                
                % 特徵萃取 (分析頻寬設定為 2.5 ~ 4.5 MHz)
                valid_mask = (f_axis >= 2.5 & f_axis <= 4.5); 
                p_fit = polyfit(f_axis(valid_mask), final_bsc(valid_mask), 1);
                
                slope = p_fit(1);       
                mbf = slope * 3.5 + p_fit(2);
                mean_bsc = mean(final_bsc(valid_mask));
                
                % ====================================================
                % 2. 論文級 Pipeline 三聯圖生成
                % ====================================================
                h_fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100, 100, 1300, 450]);
                sgtitle(sprintf('Quantitative Ultrasound Data Pipeline - %s (SNR: %.2f)', clean_save_name, target_snr), 'FontSize', 15, 'FontWeight', 'bold', 'Interpreter', 'none');
                
                % --- Panel 1: Step 1: Spatial Localization ---
                subplot(1, 3, 1);
                pcolor(X_sector, Z_sector, img_rect); shading flat; colormap(gca, gray);
                axis image; set(gca, 'YDir', 'reverse'); clim([0, 50]); hold on;
                
                % 畫出 Reference 與 Target 框，並加上文字標籤
                rectangle('Position', current_roi_r, 'EdgeColor', 'g', 'LineWidth', 2.5);
                text(current_roi_r(1), current_roi_r(2)-0.3, 'Reference', 'Color', 'g', 'FontWeight', 'bold', 'FontSize', 12);
                rectangle('Position', current_roi_t, 'EdgeColor', 'r', 'LineWidth', 2.5);
                text(current_roi_t(1), current_roi_t(2)-0.3, 'Target', 'Color', 'r', 'FontWeight', 'bold', 'FontSize', 12);
                
                title('Step 1: Spatial Localization', 'FontSize', 13, 'FontWeight', 'bold');
                xlabel('Width (cm)'); ylabel('Depth (cm)');
                
                % --- Panel 2: Step 2: Atten-Compensated Spectra ---
                subplot(1, 3, 2);
                plot(bsc_t.freq / MHz, bsc_t.PS_dB, 'r-', 'LineWidth', 2); hold on;
                plot(bsc_r.freq / MHz, bsc_r.PS_dB, 'g-', 'LineWidth', 2);
                
                % 標示我們選定的頻寬邊界 (2.5 - 4.5 MHz)
                xline(2.5, ':k', 'LineWidth', 1.5);
                xline(4.5, ':k', 'LineWidth', 1.5);
                xlim([1.5, 6]); grid on;
                
                title(sprintf('Step 2: Atten-Compensated Spectra\n(\\alpha_0 = %.2f dB/cm/MHz)', alpha_est), 'FontSize', 13, 'FontWeight', 'bold');
                xlabel('Frequency (MHz)'); ylabel('Power (dB)');
                legend({'Liver Target', 'Capsule Reference', 'Analysis Band'}, 'Location', 'southwest', 'FontSize', 10);
                
                % --- Panel 3: Step 3: Absolute BSC Feature Extraction ---
                subplot(1, 3, 3);
                plot(f_axis, final_bsc, 'b-', 'LineWidth', 2); hold on;
                
                % 畫出擬合直線
                f_valid = f_axis(valid_mask);
                plot(f_valid, polyval(p_fit, f_valid), 'r--', 'LineWidth', 3);
                
                % 畫出 Capsule Baseline (-20dB)
                yline(p.capsule_abs_bsc, 'g--', 'Capsule Baseline', 'LineWidth', 2, 'LabelHorizontalAlignment', 'center', 'LabelVerticalAlignment', 'bottom');
                
                % 標示出 MBF (在 3.5 MHz 的點)
                plot(3.5, mbf, 'r.', 'MarkerSize', 30);
                xlim([1.5, 6]); grid on;
                
                title(sprintf('Step 3: Absolute BSC Feature Extraction\nMBF: %.2f dB | Slope: %.2f', mbf, slope), 'FontSize', 13, 'FontWeight', 'bold');
                xlabel('Frequency (MHz)'); ylabel('Absolute BSC (dB)');
                legend({'Absolute BSC', 'Linear Fit (2.5-4.5 MHz)', 'MBF (3.5 MHz)'}, 'Location', 'southwest', 'FontSize', 10);
                
                % 輸出存檔
                exportgraphics(h_fig, fullfile(output_img_dir, [clean_save_name, '.png']), 'Resolution', 150);
                close(h_fig); drawnow;
                
                % ====================================================
                
                % 存入暫存陣列
                valid_frames = valid_frames + 1;
                temp_features(valid_frames, :) = [mbf, slope, mean_bsc, alpha_est, alpha_raw, target_snr];
                
            catch ME
                continue;
            end
        end
        
        % 🌟 終極核心：全幀聚合 + 擾動特徵提取 (Heterogeneity)
        if valid_frames > 0
            valid_data = temp_features(1:valid_frames, :); 
            
            % 1. 計算中位數 (保留整體趨勢)
            patient_median = median(valid_data, 1); 
            
            % 2. 計算標準差 (量化脂肪肝造成的聲學不均勻度)
            if valid_frames > 1
                patient_std = std(valid_data, 0, 1);
            else
                patient_std = zeros(1, 6); % 只有一幀時無變異數
            end
            
            % 提取特徵
            final_mbf      = patient_median(1);
            final_slope    = patient_median(2);
            final_mean_bsc = patient_median(3);
            final_est_ac   = patient_median(4);
            final_raw_ac   = patient_median(5);
            median_snr     = patient_median(6);
            
            std_mbf = patient_std(1);
            std_ac  = patient_std(4);
            std_snr = patient_std(6);
            
            fprintf('OK! [全幀聚合: %d 幀 | 中位AC: %.2f | 中位SNR: %.2f | MBF擾動(Std): %.2f]\n', valid_frames, final_est_ac, median_snr, std_mbf);
            
            new_row = {patient_name, valid_frames, final_mean_bsc, final_slope, final_mbf, final_est_ac, final_raw_ac, median_snr, std_mbf, std_ac, std_snr};
            results_table = [results_table; new_row];
        else
            fprintf('❌ 全數影像無效。\n');
        end
    end
    
    % F. 儲存目前分級的專屬 Excel 表格 (新增擾動特徵欄位)
    if ~isempty(results_table)
        results_table.Properties.VariableNames = {'Patient_ID', 'Valid_Frames_Used', 'Median_Mean_Abs_BSC', ...
            'Median_True_Spectral_Slope', 'Median_True_MBF_Value', 'Median_Unbounded_AC', 'Median_Raw_AC', ...
            'Median_SNR', 'Std_MBF', 'Std_AC', 'Std_SNR'};
        output_excel_name = ['QUS_Unbounded_Median_Results_', grade_name, '.xlsx'];
        writetable(results_table, output_excel_name);
        fprintf('💾 [%s] 完成！已導出具備擾動特徵的總表: %s\n\n', grade_name, output_excel_name);
    end
end
disp('======================================================');
disp('🎉🎉🎉【全幀異質性特徵系統 執行完畢！】🎉🎉🎉');
disp('======================================================');
end

% ========================================================================
%  底層智能核心演算法模組
% ========================================================================
function [best_roi, final_snr] = auto_find_clean_target_roi(rf_raw, n_samples, n_lines, p, angle_span)
    roi_w = 2.5; roi_h = 1.2; x_base = -roi_w / 2; z_base = 4.5 - (roi_h / 2);
    x_shifts = linspace(-0.4, 0.4, 5);  z_shifts = linspace(-0.4, 0.4, 5);
    best_roi = [x_base, z_base, roi_w, roi_h]; best_score = -1;
    
    for i = 1:length(x_shifts)
        for j = 1:length(z_shifts)
            candidate_roi = [x_base + x_shifts(i), z_base + z_shifts(j), roi_w, roi_h];
            try
                [block_rf, ~] = extract_rf_block(candidate_roi, rf_raw, n_samples, n_lines, p, angle_span);
                env = abs(hilbert(block_rf));
                mu = mean(env(:)); sigma = std(env(:)); snr = mu / sigma;
                
                sorted_env = sort(env(:)); p5_val = sorted_env(round(length(sorted_env) * 0.05));
                if (p5_val / mu) < 0.15, score = snr * 0.1; else, score = snr; end
                if score > best_score, best_score = score; best_roi = candidate_roi; end
            catch, continue; end
        end
    end
    [block_rf, ~] = extract_rf_block(best_roi, rf_raw, n_samples, n_lines, p, angle_span);
    env = abs(hilbert(block_rf)); final_snr = mean(env(:)) / std(env(:));
end

% 🎯 Reference Frequency Method (含 + eps 防呆)
function [alpha_est, alpha_raw] = estimateAC_RFM(rf_block, fs, soundspeed, freq_band_MHz)
    [n_samples, n_lines] = size(rf_block);
    idx_shallow = 1 : floor(n_samples * 0.25); 
    idx_deep = floor(n_samples * 0.5) : floor(n_samples * 0.75);
    
    z_shallow_cm = mean(idx_shallow) / fs * (soundspeed / 2) * 100; 
    z_deep_cm = mean(idx_deep) / fs * (soundspeed / 2) * 100;
    delta_z_cm = z_deep_cm - z_shallow_cm; 
    
    win_length = floor(length(idx_shallow) / 2); window = hann(win_length); noverlap = floor(win_length / 2); nfft = 1024;
    ps_shallow_avg = zeros(nfft/2 + 1, 1); ps_deep_avg = zeros(nfft/2 + 1, 1);
    
    for i = 1:n_lines
        [pxx_s, f] = pwelch(rf_block(idx_shallow, i), window, noverlap, nfft, fs);
        [pxx_d, ~] = pwelch(rf_block(idx_deep, i), window, noverlap, nfft, fs);
        ps_shallow_avg = ps_shallow_avg + pxx_s; ps_deep_avg = ps_deep_avg + pxx_d;
    end
    ps_shallow_avg = ps_shallow_avg / n_lines; ps_deep_avg = ps_deep_avg / n_lines;
    
    f_MHz = f / 1e6; mask = (f_MHz >= freq_band_MHz(1)) & (f_MHz <= freq_band_MHz(2));
    f_valid = f_MHz(mask); 
    ps_s_valid = ps_shallow_avg(mask); ps_d_valid = ps_deep_avg(mask);
    
    % 🌟 加上 eps 防止出現 -Inf 崩潰
    log_diff_dB = 10 * log10((ps_s_valid + eps) ./ (ps_d_valid + eps));
    p_fit = polyfit(f_valid, log_diff_dB, 1);
    slope_dB_MHz = p_fit(1);
    
    alpha_raw = slope_dB_MHz / (2 * delta_z_cm);
    
    alpha_est = alpha_raw;
    if alpha_est < -1.0, alpha_est = -1.0; elseif alpha_est > 3.0, alpha_est = 3.0; end
    if isnan(alpha_est) || isinf(alpha_est), alpha_est = 0.50; end
end

function [block, z_top_m] = extract_rf_block(pos, rf_raw, n_samples, n_lines, p, angle_span)
    cx = pos(1) + pos(3)/2; cz = pos(2) + pos(4)/2 + p.R0*100;
    this_r = sqrt(cx^2 + cz^2); this_th = atan2(cx, cz);
    s_idx = round(((this_r/100) - p.R0) / (p.soundspeed/2) * p.samplingrate);
    l_idx = round((this_th + angle_span/2) / (angle_span/(n_lines-1))) + 1;
    r_rng = max(1, s_idx-100) : min(n_samples, s_idx+100); l_rng = max(1, l_idx-20) : min(n_lines, l_idx+20);
    z_top_m = r_rng(1) / p.samplingrate * (p.soundspeed/2); block = rf_raw(r_rng, l_rng); 
end