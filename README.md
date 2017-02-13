# Operator Splitting QP Solver

OSQP (Operator Splitting Quadratic Program) solver is a numerical optimization package for solving problems in the form
```
minimize        1/2*x'Px + q'x
subject to      l <= A x <= u
```

The current version is `0.0.0`.

## TODO (Code)

-   [x] Fix polishing unsuccessful when solving a feasibility problem
-   [x] Check infeasibility in the code with new conditions
-   [ ] Parameter Selection: `rho`, `sigma` and `alpha` from examples (data driven)
-   [ ] **Compute cost function only when algorithm exits**. Not at every iteration!
-   [ ] Implement cheaper dual residual computation: (only one matrix-vector computation)
-   [ ] Implement functions to update `sigma`, `eps_inf` and `eps_unb` parameters
-   [ ] Stress tests Maros Meszaros
-   [ ] Link to CVXPY


## TODO (Paper)
-   [x] Write infeasibility/unbounedness conditions
-   [ ] Write compact algorithm (basic ADMM steps)
-   [ ] Talk about paper: [Inertial Proximal ADMM for Linearly Constrained Separable Convex Optimization](http://epubs.siam.org/doi/pdf/10.1137/15100463X) (similar to our robustification idea)
-   [ ] Proove convergence to vectors satisfying Farkas lemma
-   [ ] Write examples in the paper

#### Submission: [Journal of Machine Learning Research](http://www.jmlr.org/)
[Rules](http://www.jmlr.org/author-info.html#Originality) for conference submissions before journal ones:

> Submissions to JMLR cannot have been published previously in any other journal. We will consider submissions that have been published at workshops or conferences. In these cases, we expect the JMLR submission to cite the prior work, go into much greater depth and to extend the published results in a substantive way. In all cases, authors must notify JMLR about previous publication at the time of submission, and explain the differences from their prior work.
>
> We will also consider concurrent submissions of papers that are under review at conferences, provided the conference explicitly allows for this. In this case, too, we expect that the difference between the papers satisfy the above requirements, and we ask authors to provide their conference submission to the JMLR action editor in charge of the JMLR submission.
>
> Examples of (possibly) acceptable 'deltas' beyond a conference paper include: new theoretical results, entirely new application domains, significant new insights and/or analyses. Examples of insufficient deltas include: adding proofs that were omitted from a conference paper; minor variations or extensions of previous experiments; adding extra background material or references. However, we ultimately leave the decision about whether a 'delta' is significant enough up to the individual reviewers.


### Other Test Problems

-   [QPLIB2014](http://www.lamsade.dauphine.fr/QPlib2014/doku.php)
-   [Maros and Meszaros Convex Quadratic Programming](https://github.com/YimingYAN/QP-Test-Problems) Test Problem Set
