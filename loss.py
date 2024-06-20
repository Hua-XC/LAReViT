import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable

        
def comp_dist(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    #torch.Size([96, 1024])
    m, n = emb1.shape[0], emb2.shape[0]
    #a2+b2
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    #dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    #(a-b)2
    dist_mtx = dist_mtx.addmm_(emb1, emb2.t(), beta=1, alpha=-2)
    
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx   


class CenterPNLoss(nn.Module):
    def __init__(self, k_size=4, margin=0.1):
        super(CenterPNLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        
        inputsRGB=inputs[0:n//2]
        targetRGB=targets[0:n//2]
        inputsIR=inputs[n//2:n]       
        targetIR=targets[n//2:n] 
        # Come to centers
        centersRGB = []
        centersIR = []
        
        for i in range(n//2):
            centersRGB.append(inputsRGB[targetRGB == targetRGB[i]].mean(0))
            centersIR.append(inputsIR[targetIR == targetIR[i]].mean(0))
        
        #array
        centersRGB = torch.stack(centersRGB)
        centersIR = torch.stack(centersIR)
        # centers:torch.Size([96, 2048]) input：torch.Size([96, 2048])       
        dist_pc = (centersRGB - centersIR)**2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()

        centersRGB=torch.cat([centersRGB,centersRGB])
        centersIR=torch.cat([centersIR,centersIR])
        distRGB = comp_dist(centersRGB,inputs)
        distIR = comp_dist(centersIR,inputs)
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_R, dist_I = [], []
        dist_RP, dist_IP = [], []
        beta=0.1
        gama=0.1
        for i in range(0, n):
            dist_R.append(distRGB[i][mask[i] == 0].clamp(min=0.0).mean())
            dist_I.append(distIR[i][mask[i] == 0].clamp(min=0.0).mean())
            dist_RP.append(distRGB[i][mask[i]].clamp(min=0.0).min())
            dist_IP.append(distIR[i][mask[i]].clamp(min=0.0).min())
        dist_R = torch.stack(dist_R)
        dist_I = torch.stack(dist_I)
        dist_RP = torch.stack(dist_RP)
        dist_IP = torch.stack(dist_IP)
        alpha=0
        loss=dist_pc.sum()/((dist_R.sum()+dist_I.sum()-dist_pc.sum()))
        
        return loss #, dist_pc.mean(), dist_an.mean()class OriTripletLoss(nn.Module):  


class OriTripletLoss(nn.Module): 
    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):

        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(),beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss#, correct       
        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)


        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct
        
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    

def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

class DCL(nn.Module):
    def __init__(self, num_pos=4, feat_norm='no'):
        super(DCL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self,inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)
        temps=2
        N = inputs.size(0)
        id_num = N // temps // self.num_pos

        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t())
        is_neg_c2i = is_neg[::self.num_pos, :].chunk(temps, 0)[0]  # mask [id_num, N]

        centers = []
        for i in range(id_num):
            centers.append(inputs[targets == targets[i * self.num_pos]].mean(0))
        centers = torch.stack(centers)

        dist_mat = pdist_torch(centers, inputs)  #  c-i

        an = dist_mat * is_neg_c2i
        an = an[an > 1e-6].view(id_num, -1)

        d_neg = torch.mean(an, dim=1, keepdim=True)
        mask_an = (an - d_neg).expand(id_num, N - temps * self.num_pos).lt(0)  # mask
        an = an * mask_an

        list_an = []
        for i in range (id_num):
            list_an.append(torch.mean(an[i][an[i]>1e-6]))
        an_mean = sum(list_an) / len(list_an)
        #~线的意思就是翻过来true<——>false
        ap = dist_mat * ~is_neg_c2i
        ap_mean = torch.mean(ap[ap>1e-6])

        loss = ap_mean / an_mean

        return loss


class CenterTripletLoss(nn.Module):
    def __init__(self, k_size, margin=0):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        #array
        centers = torch.stack(centers)
        # centers:torch.Size([96, 2048]) input：torch.Size([96, 2048])       
        dist_pc = (inputs - centers)**2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(centers, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, centers, centers.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_an, dist_ap = [], []
        for i in range(0, n, self.k_size):
            dist_an.append((self.margin - dist[i][mask[i] == 0]).clamp(min=0.0).mean() )
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        loss = dist_pc.mean() + dist_an.mean()
        return loss/2#, dist_pc.mean(), dist_an.mean()

       
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

class MSEL(nn.Module):
    def __init__(self,num_pos=4,feat_norm = 'no'):
        super(MSEL, self).__init__()
        self.num_pos = num_pos
        self.feat_norm = feat_norm

    def forward(self, inputs, targets):
        if self.feat_norm == 'yes':
            inputs = F.normalize(inputs, p=2, dim=-1)

        target, _ = targets.chunk(2,0)
        N = target.size(0)

        dist_mat = pdist_torch(inputs, inputs)

        dist_intra_rgb = dist_mat[0 : N, 0 : N]
        dist_cross_rgb = dist_mat[0 : N, N : 2*N]
        dist_intra_ir = dist_mat[N : 2*N, N : 2*N]
        dist_cross_ir = dist_mat[N : 2*N, 0 : N]

        # shape [N, N]
        is_pos = target.expand(N, N).eq(target.expand(N, N).t())

        dist_intra_rgb = is_pos * dist_intra_rgb
        #torch.topk(tensor1, k=3, dim=1, largest=True)把tenser中的那个最大的k个拿出来
        intra_rgb, _ = dist_intra_rgb.topk(self.num_pos - 1, dim=1 ,largest = True, sorted = False) # remove itself
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        dist_intra_ir = is_pos * dist_intra_ir
        intra_ir, _ = dist_intra_ir.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = dist_cross_rgb[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_rgb = torch.mean(dist_cross_rgb, dim =1)

        dist_cross_ir = dist_cross_ir[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_ir = torch.mean(dist_cross_ir, dim=1)

        loss = (torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2)) +
                torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))) / 2

        return loss

class MSEL_Cos(nn.Module):          # for features after bn
    def __init__(self,num_pos):
        super(MSEL_Cos, self).__init__()
        self.num_pos = num_pos

    def forward(self, inputs, targets):

        inputs = nn.functional.normalize(inputs, p=2, dim=1)

        target, _ = targets.chunk(2,0)
        N = target.size(0)

        dist_mat = 1 - torch.matmul(inputs, torch.t(inputs))

        dist_intra_rgb = dist_mat[0: N, 0: N]
        dist_cross_rgb = dist_mat[0: N, N: 2*N]
        dist_intra_ir = dist_mat[N: 2*N, N: 2*N]
        dist_cross_ir = dist_mat[N: 2*N, 0: N]

        # shape [N, N]
        is_pos = target.expand(N, N).eq(target.expand(N, N).t())

        dist_intra_rgb = is_pos * dist_intra_rgb
        intra_rgb, _ = dist_intra_rgb.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)  # remove itself
        intra_mean_rgb = torch.mean(intra_rgb, dim=1)

        dist_intra_ir = is_pos * dist_intra_ir
        intra_ir, _ = dist_intra_ir.topk(self.num_pos - 1, dim=1, largest=True, sorted=False)
        intra_mean_ir = torch.mean(intra_ir, dim=1)

        dist_cross_rgb = dist_cross_rgb[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_rgb = torch.mean(dist_cross_rgb, dim=1)

        dist_cross_ir = dist_cross_ir[is_pos].contiguous().view(N, -1)  # [N, num_pos]
        cross_mean_ir = torch.mean(dist_cross_ir, dim=1)

        loss = (torch.mean(torch.pow(cross_mean_rgb - intra_mean_rgb, 2)) +
               torch.mean(torch.pow(cross_mean_ir - intra_mean_ir, 2))) / 2

        return loss

class MSEL_Feat(nn.Module):    # compute MSEL loss by the distance between sample and center
    def __init__(self, num_pos):
        super(MSEL_Feat, self).__init__()
        self.num_pos = num_pos

    def forward(self, input1, input2):
        N = input1.size(0)
        id_num = N // self.num_pos

        feats_rgb = input1.chunk(id_num, 0)
        feats_ir = input2.chunk(id_num, 0)

        loss_list = []
        for i in range(id_num):
            cross_center_rgb = torch.mean(feats_rgb[i], dim=0)  # cross center
            cross_center_ir = torch.mean(feats_ir[i], dim=0)

            for j in range(self.num_pos):

                feat_rgb = feats_rgb[i][j]
                feat_ir = feats_ir[i][j]

                intra_feats_rgb = torch.cat((feats_rgb[i][0:j], feats_rgb[i][j+1:]), dim=0)  # intra center
                intra_feats_ir = torch.cat((feats_rgb[i][0:j], feats_rgb[i][j+1:]), dim=0)

                intra_center_rgb = torch.mean(intra_feats_rgb, dim=0)
                intra_center_ir = torch.mean(intra_feats_ir, dim=0)

                dist_intra_rgb = pdist_torch(feat_rgb.view(1, -1), intra_center_rgb.view(1, -1))
                dist_intra_ir = pdist_torch(feat_ir.view(1, -1), intra_center_ir.view(1, -1))

                dist_cross_rgb = pdist_torch(feat_rgb.view(1, -1), cross_center_ir.view(1, -1))
                dist_cross_ir = pdist_torch(feat_ir.view(1, -1), cross_center_rgb.view(1, -1))

                loss_list.append(torch.pow(dist_cross_rgb - dist_intra_rgb, 2) + torch.pow(dist_cross_ir - dist_intra_ir, 2))

        loss = sum(loss_list) / N / 2

        return loss

class HXCS_contrastLoss_ori(nn.Module):

    def __init__(self, batch_size, margin=0.3):
        super(HXCS_contrastLoss_ori, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, label,T=0.015):  
        n=label.shape[0]
        #这步得到它的相似度矩阵
        similarity_matrix = F.cosine_similarity(inputs.unsqueeze(1), inputs.unsqueeze(0), dim=2)

        #试试看用距离
        #similarity_matrix=comp_dist(inputs,inputs)


        #这步得到它的label矩阵，相同label的位置为1
        mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
        
        #这步得到它的不同类的矩阵，不同类的位置为1
        mask_no_sim = torch.ones_like(mask) - mask
        
        #这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = torch.ones_like(similarity_matrix) - torch.eye(n, n).cuda()
        
        #这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = torch.exp(similarity_matrix/T)
        
        #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        similarity_matrix = similarity_matrix*mask_dui_jiao_0
        
        
        #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask*similarity_matrix
        
        
        #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim
        
        
        #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim , dim=1)
        
        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        '''
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum  = sim + no_sim_sum_expend
        loss = torch.div(sim , sim_sum)
        
        
        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        loss = mask_no_sim + loss + torch.eye(n, n).cuda()
        
        
        #接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  #求-log
        #loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n   
        
        #最后一步也可以写为---建议用这个， (len(torch.nonzero(loss)))表示一个批次中样本对个数的一半
        return torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))

class HXCS_contrastLoss(nn.Module):

    def __init__(self, batch_size, margin=0.3):
        super(HXCS_contrastLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, label,T=0.015):  
        # 计算query和key的相似度得分
        N = inputs.size(0)
        q=inputs[0:N//2]
        k=inputs[N//2:N]
        ll=label[0:N//2]
        similarity_scores = torch.matmul(q, k.t())  # 矩阵乘法计算相似度得分

        # 计算相似度得分的温度参数
        temperature = 0.015

        # 计算logits
        logits = similarity_scores / temperature

        # 构建labels（假设有N个样本）
        N = q.size(0)
        labels = torch.arange(N).to(logits.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)

        return loss

class HXCD_contrastLoss(nn.Module):

    def __init__(self, batch_size, margin=0.3):
        super(HXCD_contrastLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, label,T=100):
    
        nn=label.shape[0]
        input1=inputs[0:nn//2]
        input2=inputs[nn//2:nn]
        label1=label[0:nn//2]
        n=label.shape[0]
        


        #试试看用距离
        similarity_matrix=-pdist_torch(inputs,inputs)


        #这步得到它的label矩阵，相同label的位置为1
        mask_p = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
        
        #这步得到它的不同类的矩阵，不同类的位置为1
        mask_n = torch.ones_like(similarity_matrix) * (label.expand(n, n).ne(label.expand(n, n).t()))
        
        #这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dj0 = torch.ones_like(similarity_matrix) - torch.eye(n, n).cuda()
        
        #这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = torch.exp(similarity_matrix/T)
        
        #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        similarity_matrix = similarity_matrix*mask_dj0
        
        
        #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim_p = mask_p*similarity_matrix
        
        
        #这步产生了相不同类别的相似度矩阵
        sim_n = mask_n*similarity_matrix
        
        
        #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        sim_n_sum = torch.sum(sim_n , dim=1)
        
        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        '''
        #全体的分母，每个样本的分母都是不一样的
        sim_n_sum_expend = sim_n_sum.repeat(n, 1).T
        sim_sum  = sim_p + sim_n_sum_expend
        loss = torch.div(10*sim_p , sim_sum)
        
        
        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        #loss = mask_n + loss + torch.eye(n, n).cuda()
        loss=loss#+torch.ones_like(loss)
        
        
        #接下来就是算一个批次中的loss了
        loss = -1*torch.log(torch.sum(loss, dim=1))  #求-log
        #loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n
        
        
        #最后一步也可以写为---建议用这个， (len(torch.nonzero(loss)))表示一个批次中样本对个数的一半
        return torch.sum(loss)  
    
class HXCS_contrastLoss(nn.Module):

    def __init__(self, batch_size, margin=0.3):
        super(HXCS_contrastLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, label,T=0.015):  
        n=label.shape[0]
        #这步得到它的相似度矩阵
        similarity_matrix = F.cosine_similarity(inputs.unsqueeze(1), inputs.unsqueeze(0), dim=2)

        #试试看用距离
        #similarity_matrix=comp_dist(inputs,inputs)


        #这步得到它的label矩阵，相同label的位置为1
        mask = torch.ones_like(similarity_matrix) * (label.expand(n, n).eq(label.expand(n, n).t()))
        
        #这步得到它的不同类的矩阵，不同类的位置为1
        mask_no_sim = torch.ones_like(mask) - mask
        
        #这步产生一个对角线全为0的，其他位置为1的矩阵
        mask_dui_jiao_0 = torch.ones_like(similarity_matrix) - torch.eye(n, n).cuda()
        
        #这步给相似度矩阵求exp,并且除以温度参数T
        similarity_matrix = torch.exp(similarity_matrix/T)
        
        #这步将相似度矩阵的对角线上的值全置0，因为对比损失不需要自己与自己的相似度
        similarity_matrix = similarity_matrix*mask_dui_jiao_0
        
        
        #这步产生了相同类别的相似度矩阵，标签相同的位置保存它们的相似度，其他位置都是0，对角线上也为0
        sim = mask*similarity_matrix
        
        
        #用原先的对角线为0的相似度矩阵减去相同类别的相似度矩阵就是不同类别的相似度矩阵
        no_sim = similarity_matrix - sim
        
        
        #把不同类别的相似度矩阵按行求和，得到的是对比损失的分母(还差一个与分子相同的那个相似度，后面会加上)
        no_sim_sum = torch.sum(no_sim , dim=1)
        
        '''
        将上面的矩阵扩展一下，再转置，加到sim（也就是相同标签的矩阵上），然后再把sim矩阵与sim_num矩阵做除法。
        至于为什么这么做，就是因为对比损失的分母存在一个同类别的相似度，就是分子的数据。做了除法之后，就能得到
        每个标签相同的相似度与它不同标签的相似度的值，它们在一个矩阵（loss矩阵）中。
        '''
        no_sim_sum_expend = no_sim_sum.repeat(n, 1).T
        sim_sum  = sim + no_sim_sum_expend
        loss = torch.div(sim , sim_sum)
        
        
        '''
        由于loss矩阵中，存在0数值，那么在求-log的时候会出错。这时候，我们就将loss矩阵里面为0的地方
        全部加上1，然后再去求loss矩阵的值，那么-log1 = 0 ，就是我们想要的。
        '''
        loss = mask_no_sim + loss + torch.eye(n, n).cuda()
        
        
        #接下来就是算一个批次中的loss了
        loss = -torch.log(loss)  #求-log
        #loss = torch.sum(torch.sum(loss, dim=1) )/(2*n)  #将所有数据都加起来除以2n   
        
        #最后一步也可以写为---建议用这个， (len(torch.nonzero(loss)))表示一个批次中样本对个数的一半
        return torch.sum(torch.sum(loss, dim=1)) / (len(torch.nonzero(loss)))

class CenterLoss(nn.Module):
    def __init__(self, k_size, margin=0.1):
        super(CenterLoss, self).__init__()
        self.margin = margin
        self.k_size = k_size
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        # Come to centers
        centers = []
        for i in range(n):
            centers.append(inputs[targets == targets[i]].mean(0))
        #array
        centers = torch.stack(centers)
        # centers:torch.Size([96, 2048]) input：torch.Size([96, 2048])       
        dist_pc = (inputs - centers)**2
        dist_pc = dist_pc.sum(1)
        dist_pc = dist_pc.sqrt()

        return dist_pc.mean()/2#, dist_pc.mean(), dist_an.mean()

