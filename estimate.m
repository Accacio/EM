close all;
clear all;

load('./estimation_data.mat')

modes = 2^2;
[n, N]=size(Theta);

% shuffle data
rand_perm=randperm(size(Theta, 2));
Theta=Theta(:, rand_perm)
for i=1:4
Lambdas(:,:,i)=Lambdas(:,rand_perm,i)
end

P = H(:,:,1)
s = f(:,1)
P=H(:,:,1);
invP=inv(P);

for k=1:2^n-1 % 0-th is already empty
active=nonzeros(bitget(k,1:n).*(1:n));
inactive=setdiff(1:n,active);
P_mult(active,active,k+1)=adjoint(invP(active,active)).'./det(P(inactive,inactive))*det(P);
end

%%
%= EM params
maxIter=200;
maxErr=1e-12;
% Phi0=kron([vec(H(:,:,1)).' f(:,1).'],ones(1,2^n).')+1.*rand(2^n,n^2+n);
% Phi0=[reshape(P_mult,2^n,n^2).' zeros(2^n,n) ]+0.*rand(2^n,n^2+n);
% [Phi,z,err,normErr]=EM(Theta,Lambdas(:,:,1),Phi0,modes,maxIter,maxErr);
[Phi,z,err,normErr]=EM(Theta,Lambdas(:,:,1),[],modes,maxIter,maxErr);

% [Phi I]=sort(Phi);
for i=1:modes
P_est(:,:,i)=reshape(Phi(i,1:n^2),2,2);
s_est(:,i)=reshape(Phi(i,n^2+1:end),1,n);
end
sum(P_est(:,:,1)==0)==n;
P_est(abs(P_est)<1e-5)=0;

disp(['Estimated Ps'])
disp(P_est)
disp(['Nominal Ps'])
disp(P_mult)

return
%% plot data
fig=figure;
for lambda_idx=1:2
subplot(1,2,lambda_idx)
scatter3(Theta(1,:),Theta(2,:),Lambdas(lambda_idx,:,1),'k')
title(['$\lambda_' num2str(lambda_idx) '$'],'interpreter','latex')
end

%% final plot
fig=figure;
for lambda_idx=1:2
subplot(1,2,lambda_idx)
z_colors={'r','b','k','g'};
for idx_z=1:4;
scatter3(Theta(1,z==idx_z),Theta(2,z==idx_z),Lambdas(lambda_idx,z==idx_z,1),z_colors{idx_z})
hold on
end
hold off
title(['$\lambda_' num2str(lambda_idx) '$'],'interpreter','latex')
% legend(["Nominal","Modified"])
end

%%

Theta2=Theta(:,z==2).';
z_idx=convhull(Theta2);
Theta2(z_idx,:).'
return
figure
[x y] = meshgrid(-10:1:10);
hold on
for i=1:modes
f_val = P_est(1,1,i).*x + P_est(1,2,i).*y + s_est(1,i);
surf(x,y,f_val,'FaceColor','none','EdgeColor',z_colors{i});
end
hold off
