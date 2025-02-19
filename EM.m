function [Phi,z,residues,norm_err] = EM(X,Y,phi_init,modes,maxIter,maxErr)
    Pi=repmat(1/modes,1,modes);

    k = modes;
    [n, N] = size(Y);
    z = zeros(1,N);

    if(isempty(phi_init))
        Phi=20*rand(modes,n^2+n);
    else
        Phi=phi_init;
    end
    Phi_new=Phi;
    OldPhi=zeros(size(Phi));
    OldclusterSize=zeros(1,modes);

    Sigma_amp=100;
    Sigma(:,:,1:modes)=repmat(Sigma_amp*eye(n),1,1,modes);

    exit_msg=0;

    for j=1:maxIter
        %= kPC Cluster assignement
        Upsilon=kron(ones(n,1),eye(n));
        Delta=kron(eye(N),ones(1,n));
        G=kron(ones(1,N),eye(n));
        Y_=kron(G,ones(n,1));
        Omega=[(Upsilon*X*Delta).*Y_; G].';
        y_cell = cellfun(@(x) x',mat2cell(Y',ones(1,N))','UniformOutput',0);
        y_diag=blkdiag(y_cell{:});

        % calculate errors
        for i=1:modes
            y_lin=reshape(Omega*-Phi(i,:)',n,N);
            y_lin_cell = cellfun(@(x) x',mat2cell(y_lin',ones(1,N))','UniformOutput',0);
            y_lin_diag=blkdiag(y_lin_cell{:});
            err_diag=y_diag-y_lin_diag;
            err_cell=y_lin_cell-y_cell;
            SigmaInvMat=kron(sparse(eye(N)),sparse(Sigma(:,:,i)\eye(n)));

            respNum(:,i)=Pi(1,i)*(1/(2*pi)^(n/2))*(1/sqrt(det(Sigma(:,:,i))))*exp(-1/2*max(err_diag'*SigmaInvMat*err_diag))';
        end
        normalizing_constant = sum(respNum,2);
        Responsibilities = (respNum./normalizing_constant)';

        %= Cluster Update
        pi_new(:)=sum(Responsibilities,2)/N;
        for i=1:modes
            responsibilities=Responsibilities(i,:);
            resp2=cellfun(@(x) x*eye(n),mat2cell(responsibilities',ones(1,N)),'UniformOutput',0);
            Gamma=sqrt(sparse(blkdiag(resp2{:})));

            Phi_new(i,:)=-(Gamma*Omega)\(Gamma*Y(:));

            y_lin_new=reshape(Omega*-Phi(i,:)',n,N);
            err_lin(:,:,i)=Y-y_lin_new;
        end

        Responsibilities_sums=sum(Responsibilities,2);

        for i=1:modes
            err_lin_cell(:,:,i) = cellfun(@(x) x',mat2cell(err_lin(:,:,i)',ones(1,N))','UniformOutput',0);
            err_lin_mult=cellfun(@(x,y) x.'*y,err_lin_cell(:,:,i),err_lin_cell(:,:,i));
            Sigma_new_i=sum(Responsibilities(i,:).*err_lin_mult)/Responsibilities_sums(i)
            Sigma_new(:,:,i)=eye(n)*Sigma_new_i;
        end

        [~,z]=max(Responsibilities,[],1);

        residues=Y(:)-Omega*Phi_new.';
        norm_err=norm(residues,'fro');

        if norm_err/norm(Y,'fro')<maxErr
            exit_msg=1;
            break;
        end

        if max(sum((Phi_new-Phi).^2,2))<maxErr
            exit_msg=2;
            break;
        end

        % OldclusterSize=clusterSize;
        Phi=Phi_new;
        Sigma=Sigma_new;
    end

    switch exit_msg
    case 1
      disp(['maxErr residues'])
    case 2
        disp(['minimal iteration on estimate'])
    otherwise
      disp(['max iter reached'])
    end

end
