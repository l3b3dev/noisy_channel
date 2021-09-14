% Demo: learn a nonlinear channel with an abrupt change. Run and compare
% all algorithms using their default parameters.
%
% This file is part of the Kernel Adaptive Filtering Toolbox for Matlab.
% https://github.com/steven2358/kafbox/

close all
clear
rng('default'); rng(1); % for reproducibility

%% PARAMETERS

N = 5000; % number of training data
N_test = 300; % number of test data
fun = 'x+0.25*x.*x+0.11*x.*x.*x'; % non-linearity
SNR = 15; % SNR in dB
sigpower = 0.5^2; % input signal power

%% PROGRAM

w = who;
for i=1:length(w) % copy all parameters to option structure
    eval(sprintf('opt.%s = %s;',w{i},w{i}))
end

% generate data
[X,y,y_ref,X_test,y_test] = generate_channel_data(opt);

n = 0:N-1;
figure; 
%set(gcf, 'Units', 'Inches', 'Position',[0 0 2.5 1.2])
stem(n, y_ref); title('Input Signal')
xlabel('Samples'); ylabel('Amplitude');
% 
% 
% 
% MSEsig = zeros(N,18);
% MSE_final_sig = zeros(1,18);
% perform kernel search for qklsm
% j = 0;
% for sigma=0.8:0.1:2.6 
%     t1 = tic;
%     j = j+1;
%     kaf = qklms(struct('kernelpar',sigma)); 
%     for i=1:N
%         if ~mod(i,floor(N/10)), fprintf('.'); end
%         
%         y_est = kaf.evaluate(X_test);
%         MSEsig(i,j) = mean((y_test-y_est).^2);
%         
%         kaf.train(X(i,:),y(i));
%     end
%     MSE_final_sig(j) = 10*log10(mean(MSEsig(N-500:N,j)));
%     
%     fprintf(' %.2fs. Final MSE=%.2fdB\n',toc(t1),MSE_final_sig(j))
% end
% sigmas = 0.8:0.1:2.6;
% figure;plot(sigmas,MSE_final_sig,'-x')


% algorithms = {'lms','klms','qklms'};
% % perform online learning for each algorithm
% fprintf('\n')
% num_alg = length(algorithms);
% titles = cell(num_alg,1);
% MSE = zeros(N,num_alg);
% MSE_final = zeros(1,num_alg);
% for algo_ind=1:num_alg
%     t1 = tic;
%     algorithm = algorithms{algo_ind};
%     fprintf('%2d. %9s: ',algo_ind,upper(algorithm));
%     titles{algo_ind} = strrep(upper(algorithm),'_','\_');
%     
%     kaf = feval(algorithm);
%     for i=1:N
%         if ~mod(i,floor(N/10))
%             fprintf('.'); 
%         end
%         
%         y_est = kaf.evaluate(X_test);
%         MSE(i,algo_ind) = mean((y_test-y_est).^2);
%         
%         kaf.train(X(i,:),y(i));
%     end
%     MSE_final(algo_ind) = mean(MSE(N-500:N,algo_ind));
%     
%     fprintf(' %.2fs. Final MSE=%.2fdB\n',toc(t1),...
%         10*log10(MSE_final(algo_ind)))
% end
% fprintf('\n');
% 
% %% OUTPUT
% 
% % plot results in different "leagues"
% [MSE_final_sorted,ind] = sort(MSE_final,'descend');
% num_fig = ceil(num_alg/5);
% 
% remaining = num_alg;
% for fig_ind=num_fig:-1:1
%     figure; hold all
%     rm = rem(remaining,5);
%     num_in_league = (rm==0)*5 + rm;
%     % plot the results for the num_in_league worst results
%     league_inds = num_alg-remaining+num_in_league:-1:num_alg-remaining+1;
%     for i=league_inds
%         plot(10*log10(MSE(:,ind(i))),'LineWidth',1)
%     end
%     title(sprintf('League %d',fig_ind))
%     legend(titles(ind(league_inds)))
% 
%     axis([0 N 5*floor(min(10*log10(MSE(:)))/5) 0]);
%     remaining = remaining - num_in_league;
% end



algorithms = {'qklms'};
num_alg = length(algorithms);
titles = cell(num_alg,1);

Net_size = zeros(N);
for algo_ind=1:num_alg
   
    algorithm = algorithms{algo_ind};
    fprintf('%2d. %9s: ',algo_ind,upper(algorithm));
    titles{algo_ind} = strrep(upper(algorithm),'_','\_');
    
    kaf = feval(algorithm, struct('kernelpar',1.2, 'epsu',1));
    for i=1:N
        Net_size(i) = length(kaf.alpha);
        if ~mod(i,floor(N/10))
            fprintf('.'); 
        end
        
        y_est = kaf.evaluate(X_test);
      
        
        kaf.train(X(i,:),y(i));
    end
end
fprintf('\n');

%figure; hold all
plot(Net_size,'LineWidth',1)
% title(sprintf('League %d',fig_ind))
% legend(titles(ind(league_inds)))
