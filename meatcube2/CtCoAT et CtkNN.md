
**CtCoAT**

$$CtCoAT_{\theta}(x) = argmin_{y\in Y} E_{\theta}^{CoAT}(x,y),$$

where $E_\theta^{CoAT}$ is an energy function defined by

$$
E_\theta^{CoAT}(x,y) = \Gamma(\sigma_S,\sigma_R,CB\cup\{(x,y)\}) - \Gamma(\sigma_S,\sigma_R,CB)
$$

with

$$
\Gamma(\sigma_S,\sigma_R,CB) =
\sum_{i,j,k}\frac{1 - [\sigma_S(s_i,s_j)-\sigma_S(s_i,s_k)][\sigma_R(r_i,r_j)-\sigma_R(r_i,r_k)]}{2}
$$ 

**CtkNN**

The kNN decision can be expressed as:
$$
kNN( s_ t) = argmax_{ r\in\mathcal{R}}
    \left(
    \displaystyle\sum_{( s_i, r_i)\in CB}\sigma_S( s_i, s_t)\cdot\sigma_R( r_i, r)
    \right)
    $$ 
where $\sigma_S = 1_{N_k}$.

The $kNN$ decision rule can be written 
$$
kNN( s_t) = argmin_{ r\in \mathcal{R}} E_{\theta}^{kNN}( s_t, r)
$$ 
with (**pair-based version**):
$$
E_{\theta}^{kNN}( s_ t, r)=1 - \frac{1}{k}\sum_{( s_i, r_i)\in CB}1_{N_k}( s_i, s_ t).\sigma_R( r_i, r)
$$
    
With $1_{N_k}( s_i, s_ t)=1$ if $s_ t$ in the $k$NN of $s_i$, and $1_{N_k}( s_i, s_ t)=0$.



**Checking CtCoAT implementation**

Suppose $CB$ contains only two cases, i.e.,
$$
CB = \{ ◌, □ \}.
$$

Then, for a new case $Z$, ie $E_\theta^{CoAT}(Z)$ writes :
$$
\begin{align*}
    E_\theta^{CoAT}(Z)
    =
    &\displaystyle\sum_{Z_i,Z_j\in CB}
    \gamma(Z,Z,Z) + \gamma(Z_i,Z,Z) + \gamma(Z,Z,Z_i)+\gamma(Z,Z_i,Z)\\
    &+\gamma(Z,Z_i,Z_j) + \gamma(Z_i,Z,Z_j) + \gamma(Z_i,Z_j,Z)  \\\\
    =&\gamma(Z,Z,Z) + \gamma(◌,Z,Z),+ \gamma(□,Z,Z)\\
    &+ \gamma(Z,Z,□) + \gamma(Z,□,Z)
     + \gamma(Z,Z,◌) + \gamma(Z,◌,Z)\\
    &+\gamma(Z,□,◌) + \gamma(Z,◌,□)
    +\gamma(Z,□,□) + \gamma(Z,◌,◌)\\
    &+ \gamma(◌,Z,□) + \gamma(◌,□,Z)
    + \gamma(□,Z,◌) + \gamma(□,◌,Z)\\
   & + \gamma(□,Z,□) +\gamma(□,□,Z) +\gamma(◌,Z,◌)+ \gamma(◌,◌,Z)\\\\
    =& 19/2\\
    &-\left[\sigma_S(Z,Z)-\sigma_S(Z,□)\right]\left[\sigma_R(Z,Z)-\sigma_R(Z,□)\right]\\
    &-\left[\sigma_S(Z,Z)-\sigma_S(Z,◌)\right]\left[\sigma_R(Z,Z)-\sigma_R(Z,◌)\right]\\
    &-\left[\sigma_S(Z,◌)-\sigma_S(Z,□)\right]\left[\sigma_R(Z,◌)-\sigma_R(Z,□)\right]\\
    & -\left[\sigma_S(◌,Z)-\sigma_S(◌,□)\right]\left[\sigma_R(◌,Z)-\sigma_R(◌,□)\right]\\
    & -\left[\sigma_S(□,Z)-\sigma_S(□,◌)\right]\left[\sigma_R(□,Z)-\sigma_R(□,◌)\right]\\
    & -\left[\sigma_S(□,Z)-\sigma_S(□,□)\right]\left[\sigma_R(□,Z)-\sigma_R(□,□)\right]\\
    & -\left[\sigma_S(◌,Z)-\sigma_S(◌,◌)\right]\left[\sigma_R(◌,Z)-\sigma_R(◌,◌)\right]\\
\end{align*}
$$
So if $Z=(X,orange)$, 
    $$
    E_{\theta_\mathcal{T}}(Z)=15/2 
    + 3\,\sigma_S(X,◌) 
    - 2\,\sigma_S(X,□)
    +\sigma_S(◌,□)
    $$
and if $Z=(X,blue)$,
    $$
    E_{\theta_\mathcal{T}}(Z)=15/2 
    + 3\,\sigma_S(X,□) 
    - 2\,\sigma_S(X,◌)
    +\sigma_S(◌,□)
    $$

Furthermore, we have :
For $Z=(X,orange)$, 
    $$
    \ell_{hinge}(Z)=max(0,\lambda + 5\,(\sigma_S(X,◌)-\sigma_S(X,□)))
    $$
For $Z=(X,blue)$,
    $$
    \ell_{hinge}(Z)=max(0,\lambda +  5\,(\sigma_S(X,□)-\sigma_S(X,◌)))
    $$
so for all $Z=(X,Y)$,
$$
\ell_{hinge}(Z)=max(0,\lambda -  5\left[\sigma_S(X,□)-\sigma_S(X,◌)\right]\left[(\sigma_R(Y,□)-\sigma_R(Y,◌)\right])
$$
soit
$$
\mathcal{L}_{hinge}(\mathcal{T})
    =\frac{1}{|T|}
    \displaystyle\sum_{Z=(X,Y)\in\mathcal{T}}
    max(0,\lambda -  5\left[\sigma_S(X,□)-\sigma_S(X,◌)\right]\left[\sigma_R(X,□)-\sigma_R(X,◌)\right])
$$