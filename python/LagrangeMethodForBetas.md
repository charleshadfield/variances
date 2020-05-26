# Some remarks on optimizing the variance with iterative updates

We can update $\beta_{i,Q_i}$ iteratively by
$$
\beta_{i,Q_i}^{(t)} = (1 - \Delta) \beta_{i,Q_i}^{(t-1)}  + \Delta~\beta_{i,Q_i}^{\text{new}},
$$
where the new beta is obtained from the closed form equalities of the Lagrange equations. 

## Earlier cost function
Suppose that we want to optimize the following function

$$
f(\{\beta_{i, Q_i}\}_i) = \sum_{Q} \alpha_Q^2 \prod_{j \in \text{support}{(Q})} \beta^{-1}_{j, Q_j}
$$

By Lagrange multiplier method, the above function is optimized when the following equations hold: 
$$
\beta_{i,X} = \left( \sum_{Q:Q_i = X} \alpha_Q^2 \prod_{j \in \text{support}(Q)} \beta_{j, Q_j}^{-1} \right) / \left( \sum_{Q:Q_i \neq I} \alpha_Q^2 \prod_{j \in \text{support}{(Q})} \beta^{-1}_{j, Q_j} \right),
$$

and 

$$
\beta_{i,Y} = \left( \sum_{Q:Q_i = Y} \alpha_Q^2 \prod_{j \in \text{support}(Q)} \beta_{j, Q_j}^{-1} \right) / \left( \sum_{Q: Q_i \neq I} \alpha_Q^2 \prod_{j \in \text{support}{(Q})} \beta^{-1}_{j, Q_j} \right),
$$

and 

$$
\beta_{i,Z} = \left( \sum_{Q:Q_i = Z} \alpha_Q^2 \prod_{j \in \text{support}(Q)} \beta_{j, Q_j}^{-1} \right) / \left( \sum_{Q: Q_i \neq I} \alpha_Q^2 \prod_{j \in \text{support}{(Q})} \beta^{-1}_{j, Q_j} \right).
$$

The above equations can also be written as

$$
\beta_{i,X}^2 = \left( \sum_{Q:Q_i = X} \alpha_Q^2 \prod_{j \in \text{support}(Q): j \neq i} \beta_{j, Q_j}^{-1} \right) / \left( \sum_{Q: Q_i \neq I} \alpha_Q^2 \prod_{j \in \text{support}{(Q})} \beta^{-1}_{j, Q_j} \right),
$$

and

$$
\beta_{i,Y}^2 = \left( \sum_{Q:Q_i = Y} \alpha_Q^2 \prod_{j \in \text{support}(Q): j \neq i} \beta_{j, Q_j}^{-1} \right) / \left( \sum_{Q: Q_i \neq I} \alpha_Q^2 \prod_{j \in \text{support}{(Q})} \beta^{-1}_{j, Q_j} \right),
$$

and

$$
\beta_{i,Z}^2 = \left( \sum_{Q:Q_i = Z} \alpha_Q^2 \prod_{j \in \text{support}(Q): j \neq i} \beta_{j, Q_j}^{-1} \right) / \left( \sum_{Q: Q_i \neq I} \alpha_Q^2 \prod_{j \in \text{support}{(Q})} \beta^{-1}_{j, Q_j} \right),
$$

## Another cost function

Suppose that we want to optimize the following function

$$
f(\{\beta_{i, Q_i}\}_i) = \sum_{Q,R \in \mathcal{I}_{\text{comp}}} \alpha_Q \alpha_R \prod_{j:Q_j = R_j \neq I} \beta^{-1}_{j, Q_j} \prod_{j:Q_j \neq R_j} m_j.
$$

By analogy, using the Langrange multiplier, at optimality the following 
must hold:

$$
\beta_{i,X} = \left( \sum_{Q,R \in \mathcal{I}_{\text{comp}}~\text{and}~Q_i = R_i = X} \alpha_Q \alpha_R \prod_{j:Q_j = R_j \neq I} \beta^{-1}_{j, Q_j} \prod_{j:Q_j \neq R_j} m_j\right) / \left( \sum_{Q,R \in \mathcal{I}_{\text{comp}}~\text{and}~Q_i = R_i \neq I} \alpha_Q \alpha_R \prod_{j:Q_j = R_j \neq I} \beta^{-1}_{j, Q_j} \prod_{j:Q_j \neq R_j} m_j\right),
$$
and
$$
\beta_{i,Y} = \left( \sum_{Q,R \in \mathcal{I}_{\text{comp}}~\text{and}~Q_i = R_i = Y} \alpha_Q \alpha_R \prod_{j:Q_j = R_j \neq I} \beta^{-1}_{j, Q_j} \prod_{j:Q_j \neq R_j} m_j\right) / \left( \sum_{Q,R \in \mathcal{I}_{\text{comp}}~\text{and}~Q_i = R_i \neq I} \alpha_Q \alpha_R \prod_{j:Q_j = R_j \neq I} \beta^{-1}_{j, Q_j} \prod_{j:Q_j \neq R_j} m_j\right),
$$
and
$$
\beta_{i,Z} = \left( \sum_{Q,R \in \mathcal{I}_{\text{comp}}~\text{and}~Q_i = R_i = Z} \alpha_Q \alpha_R \prod_{j:Q_j = R_j \neq I} \beta^{-1}_{j, Q_j} \prod_{j:Q_j \neq R_j} m_j\right) / \left( \sum_{Q,R \in \mathcal{I}_{\text{comp}}~\text{and}~Q_i = R_i \neq I} \alpha_Q \alpha_R \prod_{j:Q_j = R_j \neq I} \beta^{-1}_{j, Q_j} \prod_{j:Q_j \neq R_j} m_j\right)
$$

