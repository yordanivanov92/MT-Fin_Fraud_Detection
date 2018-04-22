+ For $g=1$ to 2 do:
  
  - Compute negative gradient:
  $$z_{igm}=-\frac{\partial{\Psi(y_{ig},f_{g}(\mathbf{x}_i))}}{\partial{f_{g}(\mathbf{x}_i)}}\rvert_{f_{g}(\mathbf{x}_i)=\widehat{f_{g}}(\mathbf{x}_i)},\; i=1,...,N$$
    - Randomly select $p \times N$ subset.
  - Fit regression tree with K number of terminal nodes on the previously selected subset, $g(\mathbf{x})=E(z|\mathbf{x})$.
  - Compute the optimal terminal node predictions, $\rho_{1g},...,\rho_{Kg}$, as:
    $$\rho_{kgm}=\arg\min_{\rho}\sum_{\mathbf{x}_i \in S_k} \Psi(y_i,\widehat{f_{g}}(\mathbf{x}_{i})+\rho)$$,
    where $S_k$ is the set of $\mathbf{x}$'s that define terminal node $k$.
    - Update $\widehat{f_{g}}(\mathbf{x})$ as $\widehat{f_{g}}(\mathbf{x}) \leftarrow \widehat{f_{g}}(\mathbf{x}) + \lambda\rho_{k(\mathbf{x})}$, where $k(\mathbf{x})$ shows the index of the terminal node into which an observation with features $\mathbf{x}$ should fall.'
    
    
