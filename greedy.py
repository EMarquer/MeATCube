
import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import euclidean, cosine, correlation, braycurtis, hamming
def euclidean_sim(x1,x2):
    return np.exp(-euclidean(x1,x2))
def cosine_sim(x1,x2):
    return np.exp(-cosine(x1,x2))
def correlation_sim(x1,x2):
    return np.exp(-correlation(x1,x2))
def hamming_sim(x1,x2):
    return np.exp(-hamming(x1,x2))
def radius_sim(x1,x2):
    return np.exp(-abs(norm(x1)-norm(x2)))
def class_equality_sim(y1,y2):
    return np.equal(y1,y2).astype(float)






X_cb = np.array([[-1,0],[1,0]])
y_cb = np.array([1,0])

manual_energy_ = dict()

manual_energy_["CB X"] = X_cb
manual_energy_["CB y"] = y_cb

N = 10
np.random.seed(42)
X = np.random.rand(N,2)
y = np.random.randint(2,size=N)
manual_energy_["X"] = X
manual_energy_["y"] = y




CtCoAT_energy_fadi = lambda x, y: (
    ((7.5 + 3*euclidean_sim(x,x2=X_cb[1]) - 2*euclidean_sim(x,X_cb[0]) + euclidean_sim(X_cb[0],X_cb[1])) / 27)
    if y==1 else
    ((7.5 + 3*euclidean_sim(x,X_cb[0]) - 2*euclidean_sim(x,X_cb[1]) + euclidean_sim(X_cb[0],X_cb[1])) / 27)
)
CtCoAT_energy_esteban = lambda x, y: (
    ((9.5 - 5*euclidean_sim(X_cb[0],x2=X_cb[1]) + 2*euclidean_sim(X_cb[1],x) - 3 * euclidean_sim(X_cb[0],x)))
    if y==1 else
    ((9.5 - 5*euclidean_sim(X_cb[0],x2=X_cb[1]) + 2*euclidean_sim(X_cb[0],x) - 3 * euclidean_sim(X_cb[1],x)))
)
def CtCoAT_gamma(simS, simR, X_cb, y_cb):
    # Γ(σS,σR,CB)= ∑_(i,j,k) (1 - [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)])/2

    sum_result = 0
    for X_i, y_i in zip(X_cb, y_cb):
        for X_j, y_j in zip(X_cb, y_cb):
            for X_k, y_k in zip(X_cb, y_cb):
                sum_result += (1 - (simS(X_i, X_j) - simS(X_i, X_k)) * (simR(y_i, y_j) - simR(y_i, y_k)))/2
    return sum_result
def CtCoAT_energy(X_, y_):
    Gamma_CB = CtCoAT_gamma(euclidean_sim, class_equality_sim, X_cb, y_cb)
    Gamma_CB_t = CtCoAT_gamma(euclidean_sim, class_equality_sim, 
                                np.append(X_cb, values=X_.reshape(1, len(X_)), axis=0), np.append(y_cb, y_.reshape(1), axis=0))
    return Gamma_CB_t - Gamma_CB

print("\nfadi   ", [
        CtCoAT_energy_fadi(X_, y_)
        for X_, y_ in zip(X, y)
    ])
print("esteban", [
        CtCoAT_energy_esteban(X_, y_)
        for X_, y_ in zip(X, y)
    ])
print("greedy ", [
        CtCoAT_energy(X_, y_)
        for X_, y_ in zip(X, y)
    ])



"""
Result:
fadi    [0.27994193937262896, 0.2621265952972144, 0.3068184647610277, 0.2948228771105609, 0.3192225628037711, 0.292666363373946, 0.3558597775700494, 0.2843640912031346, 0.3111124276134862, 0.3242935019141348]
esteban [8.237959365046098,   7.586430325183196,  8.736738574134879,  8.615300681892762,  9.189684425293324,  8.593353340910776, 9.875122629679492,  8.13104118820996,   8.92452654435122,   9.183810048572813]
greedy  [7.558432363060981,   7.077418073024789,  8.284098548547746,  7.960217681985145,  8.619009195701821,  7.901991811096543, 9.608213994391331,  7.677830462484635,  8.400035545564126,  8.755924551681638]
"""




"""
E CoAT(x,y)=Γ(σS,σR,CB∪{(x,y)}) - Γ(σS,σR,CB)
for 
Γ(σS,σR,CB)= ∑_(i,j,k) (1 - [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)])/2
            =  1/2*(∑_(i,j,k) (1 - [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)]))
            =  1/2*(∑_(i,j,k) 1 - ∑_(i,j,k) ([σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)]))
            = (|CB|**3)/2 - 1/2*∑_(i,j,k) ([σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)]))

so for a |CB| of 2

Γ(σS,σR,CB) = 8/2 - 1/2*(
    [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 0, j = 0, k = 0
    [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 0, j = 0, k = 1
    [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 0, j = 1, k = 0
    [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 0, j = 1, k = 1
    [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 1, j = 0, k = 0
    [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 1, j = 0, k = 1
    [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)] +         i = 1, j = 1, k = 0
    [σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)]           i = 1, j = 1, k = 1
)
            = 8/2 - 1/2*(
    [σS(s0,s0) - σS(s0,s1)]*[σR(r0,r0)-σR(r0,r1)] +         i = 0, j = 0, k = 1
    [σS(s0,s1) - σS(s0,s0)]*[σR(r0,r1)-σR(r0,r0)] +         i = 0, j = 1, k = 0
    [σS(s1,s0) - σS(s1,s1)]*[σR(r1,r0)-σR(r1,r1)] +         i = 1, j = 0, k = 1
    [σS(s1,s1) - σS(s1,s0)]*[σR(r1,r1)-σR(r1,r0)] +         i = 1, j = 1, k = 0
    0 +         i = 0, j = 0, k = 0
    0 +         i = 0, j = 1, k = 1
    0 +         i = 1, j = 0, k = 0
    0           i = 1, j = 1, k = 1
)
            = 8/2 - 1/2*(
    [1 - σS(s0,s1)]*[1-σR(r0,r1)] +         i = 0, j = 0, k = 1
    [σS(s0,s1) - 1]*[σR(r0,r1)-1] +         i = 0, j = 1, k = 0
    [σS(s1,s0) - 1]*[σR(r1,r0)-1] +         i = 1, j = 0, k = 1
    [1 - σS(s1,s0)]*[1-σR(r1,r0)]           i = 1, j = 1, k = 0
)
            = 8/2 - 1/2*(
    [1 - σS(s0,s1)]*[1-σR(r0,r1)] +         i = 0, j = 0, k = 1
    -[1-σS(s0,s1)]*-[1-σR(r0,r1)] +         i = 0, j = 1, k = 0
    -[1-σS(s1,s0)]*-[1-σR(r1,r0)] +         i = 1, j = 0, k = 1
    [1 - σS(s1,s0)]*[1-σR(r1,r0)]           i = 1, j = 1, k = 0
)
            = 8/2 - 1/2*4*([1 - σS(s0,s1)]*[1-σR(r0,r1)])
                                            as σR(r1,r0) = σR(r0,r1) and similarly for σS
)
= 4 - 2*(
    1 +
    -σR(r0,r1) +
    -σS(s0,s1) +
    σS(s0,s1)*σR(r0,r1)
)
Γ(σS,σR,CB) = 4 - 2 + 2*σR(r0,r1) + 2*σS(s0,s1) - 2*σS(s0,s1)*σR(r0,r1)
for |CB|={(s0,r0), (s1,r1)}

in particular for our case using class equality for σR:
Γ(σS,σR,CB) = 2 + 2*0 + 2*σS(s0,s1) - 2*σS(s0,s1)*0 = 2 + 2*σS(s0,s1)
σS([1,0],[-1,0]) = e(-2) for Euclidean similarity
and the energy of the CB is Γ(σS,σR,CB) = 2 + 2*e(-2)
)


Now for Γ(σS,σR,CB∪{(st,rt)}):
Γ(σS,σR,CB∪{(st,rt)}) = (|CB|**3)/2 - 1/2*∑_(i,j,k) ([σS(si,sj) - σS(si,sk)]*[σR(ri,rj)-σR(ri,rk)]))
= 27/2 - 1/2*(       skipping any j=k that will result in 0
    [σS(s0,s0) - σS(s0,s1)]*[σR(r0,r0)-σR(r0,r1)] +                         i = 0, j = 0, k = 1
    [σS(s0,s1) - σS(s0,s0)]*[σR(r0,r1)-σR(r0,r0)] +                         i = 0, j = 1, k = 0
    [σS(s0,s0) - σS(s0,st)]*[σR(r0,r0)-σR(r0,rt)] +                         i = 0, j = 0, k = t
    [σS(s0,st) - σS(s0,s0)]*[σR(r0,rt)-σR(r0,r0)] +                         i = 0, j = t, k = 0
    [σS(s0,s1) - σS(s0,st)]*[σR(r0,r1)-σR(r0,rt)] +                         i = 0, j = 1, k = t
    [σS(s0,st) - σS(s0,s1)]*[σR(r0,rt)-σR(r0,r1)] +                         i = 0, j = t, k = 1
                        
    [σS(s1,s0) - σS(s1,s1)]*[σR(r1,r0)-σR(r1,r1)] +                         i = 1, j = 0, k = 1
    [σS(s1,s1) - σS(s1,s0)]*[σR(r1,r1)-σR(r1,r0)] +                         i = 1, j = 1, k = 0
    [σS(s1,s0) - σS(s1,st)]*[σR(r1,r0)-σR(r1,rt)] +                         i = 1, j = 0, k = t
    [σS(s1,st) - σS(s1,s0)]*[σR(r1,rt)-σR(r1,r0)] +                         i = 1, j = t, k = 0
    [σS(s1,s1) - σS(s1,st)]*[σR(r1,r1)-σR(r1,rt)] +                         i = 1, j = 1, k = t
    [σS(s1,st) - σS(s1,s1)]*[σR(r1,rt)-σR(r1,r1)] +                         i = 1, j = t, k = 1
                        
    [σS(st,s0) - σS(st,s1)]*[σR(rt,r0)-σR(rt,r1)] +                         i = t, j = 0, k = 1
    [σS(st,s1) - σS(st,s0)]*[σR(rt,r1)-σR(rt,r0)] +                         i = t, j = 1, k = 0
    [σS(st,s0) - σS(st,st)]*[σR(rt,r0)-σR(rt,rt)] +                         i = t, j = 0, k = t
    [σS(st,st) - σS(st,s0)]*[σR(rt,rt)-σR(rt,r0)] +                         i = t, j = t, k = 0
    [σS(st,s1) - σS(st,st)]*[σR(rt,r1)-σR(rt,rt)] +                         i = t, j = 1, k = t
    [σS(st,st) - σS(st,s1)]*[σR(rt,rt)-σR(rt,r1)]                           i = t, j = t, k = 1
)
= 27/2 - (       by symmetry of the similarity
    [σS(s0,s0) - σS(s0,s1)]*[σR(r0,r0)-σR(r0,r1)] +                         i = 0, j = 0, k = 1
    [σS(s0,s0) - σS(s0,st)]*[σR(r0,r0)-σR(r0,rt)] +                         i = 0, j = 0, k = t
    [σS(s0,s1) - σS(s0,st)]*[σR(r0,r1)-σR(r0,rt)] +                         i = 0, j = 1, k = t
                        
    [σS(s1,s0) - σS(s1,s1)]*[σR(r1,r0)-σR(r1,r1)] +                         i = 1, j = 0, k = 1
    [σS(s1,s0) - σS(s1,st)]*[σR(r1,r0)-σR(r1,rt)] +                         i = 1, j = 0, k = t
    [σS(s1,s1) - σS(s1,st)]*[σR(r1,r1)-σR(r1,rt)] +                         i = 1, j = 1, k = t
                        
    [σS(st,s0) - σS(st,s1)]*[σR(rt,r0)-σR(rt,r1)] +                         i = t, j = 0, k = 1
    [σS(st,s0) - σS(st,st)]*[σR(rt,r0)-σR(rt,rt)] +                         i = t, j = 0, k = t
    [σS(st,s1) - σS(st,st)]*[σR(rt,r1)-σR(rt,rt)]                           i = t, j = 1, k = t
)
= 27/2 - (       
    [1 - σS(s0,s1)]*[1-0] +                         i = 0, j = 0, k = 1      
    [σS(s1,s0) - 1]*[0-1] +                         i = 1, j = 0, k = 1

    [σS(st,s0) - σS(st,s1)]*[σR(rt,r0)-σR(rt,r1)] +                         i = t, j = 0, k = 1
    [σS(s0,s1) - σS(s0,st)]*[0-σR(r0,rt)] +                         i = 0, j = 1, k = t
    [σS(s1,s0) - σS(s1,st)]*[0-σR(r1,rt)] +                         i = 1, j = 0, k = t
    [1 - σS(s1,st)]*[1-σR(r1,rt)] +                         i = 1, j = 1, k = t         
    [1 - σS(s0,st)]*[1-σR(r0,rt)] +                         i = 0, j = 0, k = t
    [σS(st,s0) - 1]*[σR(rt,r0)-1] +                         i = t, j = 0, k = t
    [σS(st,s1) - 1]*[σR(rt,r1)-1]                           i = t, j = 1, k = t
)
= 27/2 - (       
    2 * [1 - σS(s0,s1)]*[1-0] +                         i = 0, j = 0, k = 1 & i = 1, j = 0, k = 1

    [σS(st,s0) - σS(st,s1)]*[σR(rt,r0)-σR(rt,r1)] +     i = t, j = 0, k = 1
    [σS(s0,s1) - σS(s0,st)]*[0-σR(r0,rt)] +             i = 0, j = 1, k = t
    [σS(s0,s1) - σS(s1,st)]*[0-σR(r1,rt)] +             i = 1, j = 0, k = t
    2 * [1 - σS(s1,st)]*[1-σR(r1,rt)] +                 i = 1, j = 1, k = t & i = t, j = 1, k = t
    2 * [1 - σS(s0,st)]*[1-σR(r0,rt)]                   i = 0, j = 0, k = t & i = t, j = 0, k = t
)

if σR(r1,rt) = 1 (i.e. rt = r1)
Γ(σS,σR,CB∪{(st,rt)}) = 27/2 - (       
    2 * [1 - σS(s0,s1)]*[1-0] +                         i = 0, j = 0, k = 1 & i = 1, j = 0, k = 1

    [σS(st,s0) - σS(st,s1)]*[0-1] +     i = t, j = 0, k = 1
    [σS(s0,s1) - σS(s0,st)]*[0-0] +             i = 0, j = 1, k = t
    [σS(s0,s1) - σS(s1,st)]*[0-1] +             i = 1, j = 0, k = t
    2 * [1 - σS(s1,st)]*[1-1] +                 i = 1, j = 1, k = t & i = t, j = 1, k = t
    2 * [1 - σS(s0,st)]*[1-0]                   i = 0, j = 0, k = t & i = t, j = 0, k = t
)
= 27/2 - (       
    2 * [1 - σS(s0,s1)] +                         i = 0, j = 0, k = 1 & i = 1, j = 0, k = 1
    -[σS(s0,st) - σS(s1,st)] +                    i = t, j = 0, k = 1
    -[σS(s0,s1) - σS(s1,st)] +                    i = 1, j = 0, k = t
    2 * [1 - σS(s0,st)]                           i = 0, j = 0, k = t & i = t, j = 0, k = t
)
= 27/2 - (       
    2 - 2 * σS(s0,s1) +                           i = 0, j = 0, k = 1 & i = 1, j = 0, k = 1
    -σS(s0,st) + σS(s1,st)] +                    i = t, j = 0, k = 1
    -σS(s0,s1) + σS(s1,st)] +                    i = 1, j = 0, k = t
    2 - 2 * σS(s0,st)                            i = 0, j = 0, k = t & i = t, j = 0, k = t
)
= 27/2 - 4 - 3 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)
= 19/2 - 3 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)

and the total energy is:

E CoAT(x,y)=Γ(σS,σR,CB∪{(x,y)}) - Γ(σS,σR,CB) 
= (27/2 - 4 - 3 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)) - (2 + 2*σS(s0,s1))
= (27/2 - 4 - 3 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)) - 2 - 2*σS(s0,s1)
= 27/2 - 6 - 5 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)
= 7.5 - 5 * σS(s0,s1) + 2 * σS(s1,st)] - 3 * σS(s0,st)
"""