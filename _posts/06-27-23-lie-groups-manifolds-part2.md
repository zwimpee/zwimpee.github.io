---
layout: post
comments: true
title:  "Applications of Lie Groups to Neural Networks - Part 2"
excerpt: "> “...different choices of local coordinate charts can satisfy the definition of a manifold. This is a key property of manifolds and is fundamental to their study in differential geometry and related fields."
date:   2023-06-27 13:00:00
mathjax: true
author: Zach Wimpee
thumbnail: /assets/lie-groups/lie-groups.jpg
---

# Applications of Lie Groups to Neural Networks - Part 2
#
## Introduction
Picking back up from where we left off in part 1 of our discussion, we will now explore the concept of a manifold in more detail. We will also explore the concept of a Lie group, which is a group that is also a manifold. We will then explore the concept of a Lie algebra, which is a vector space that is also a Lie group. Finally, we will explore the concept of a Lie algebra homomorphism, which is a linear map between two Lie algebras that preserves the Lie bracket.

## Some additional manifold examples

### Manifolds and the Sphere $$S^{2}$$
Now that we've explored the circle as a 1-dimensional manifold, let's move on to a 2-dimensional manifold: the sphere $$ S^{2} $$. We can think of the sphere as a 2-dimensional manifold because we can parameterize it using two parameters, say $$ \theta $$ and $$ \phi $$, as follows:

$$ 
x = \cos(\theta)\sin(\phi) \\
y = \sin(\theta)\sin(\phi) \\
z = \cos(\phi) 
$$

```python
# S^2 is a good example of a nontrivial two-dimensional manifold,
# realized as a surface in three-dimensional space.

# Let'a define this in terms of the subsets U_1 and U_2, which cover S^2.
# U_1 is the upper hemisphere, and U_2 is the lower hemisphere.
# We can define these subsets in terms of the following coordinate charts:
# S2 = { (x, y, z) in R^3 | x^2 + y^2 + z^2 = 1 }
# U_1 = S^2 \ { (0, 0, 1) }
# U_2 = S^2 \ { (0, 0, -1) }

tolerance = 1e-2
S2 = torch.tensor([[x.item(), y.item(), z.item()] for x in torch.linspace(-1, 1, 50) for y in torch.linspace(-1, 1, 50) for z in torch.linspace(-1, 1, 50) if abs(x**2 + y**2 + z**2 - 1.0) < tolerance ])
```

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(S2[:, 0], S2[:, 1], S2[:, 2]);
plt.show()
```

<div class="imgcap_noborder">
<img src="/assets/s2.png" width="80%">
</div>

```python
# Now we define the coordinate charts for U_1 and U_2.

# U_1 = { (x, y, z) in S^2 | z != 1 }
# U_2 = { (x, y, z) in S^2 | z != -1 }

U1 = torch.tensor([[x.item(), y.item(), z.item()] for x in torch.linspace(-1, 1, 50) for y in torch.linspace(-1, 1, 50) for z in torch.linspace(-1, 1, 50) if abs(x**2 + y**2 + z**2 - 1.0) < tolerance and abs(z - 1.0) > tolerance])
U2 = torch.tensor([[x.item(), y.item(), z.item()] for x in torch.linspace(-1, 1, 50) for y in torch.linspace(-1, 1, 50) for z in torch.linspace(-1, 1, 50) if abs(x**2 + y**2 + z**2 - 1.0) < tolerance and abs(z + 1.0) > tolerance])
```

```python
# We can plot the subsets U_1 and U_2 using the Axes3D class from the mpl_toolkits.mplot3d library.


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U1[:, 0], U1[:, 1], U1[:, 2]);
plt.show();
```

<div class="imgcap_noborder">
<img src="/assets/s2_u1.png" width="80%">
</div>

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(U2[:, 0], U2[:, 1], U2[:, 2]);
plt.show();
```

<div class="imgcap_noborder">
<img src="/assets/s2_u2.png" width="80%">
</div>

We can visually see that the subsets $$ U_1 $$ and $$ U_2 $$ cover the sphere $$ S^2 $$.

Now let's go a little bit deeper, and explicitly define the coordinate charts for $$ U_1 $$ and $$ U_2 $$, denoted $$ \chi_{\alpha} $$, where $$ \alpha \in \{1, 2\} $$.


The local coordinate maps for the sphere $$S^2$$ can be defined using spherical coordinates. For a point $$p = (x, y, z)$$ on the sphere, we can define the local coordinate maps $$\chi_{\alpha}$$ and $$\chi_{\beta}$$ as follows:

$$
\chi_{\alpha}(p) = (\theta, \phi) = (\arctan(y/x), \arccos(z))
\chi_{\beta}(p) = (\theta', \phi') = (\arctan(y/x), \pi - \arccos(z))
$$

where $$\theta, \theta' \in [0, 2\pi]$$ and $$\phi, \phi' \in [0, \pi]$$. The local coordinate maps $$\chi_{\alpha}$$ and $$\chi_{\beta}$$ map points in $$U_{\alpha}$$ and $$U_{\beta}$$ respectively to points in $$V_{\alpha} = [0, 2\pi) \times [0, \pi)$$ and $$V_{\beta} = [0, 2\pi) \times (0, \pi]$$.

We can now check the smoothness of the composite map $$\chi_{\beta} \circ \chi_{\alpha}^{-1}$$ on the overlap $$U_{\alpha} \cap U_{\beta}$$. Since $$\chi_{\alpha}$$ and $$\chi_{\beta}$$ are both smooth functions, their inverse functions $$\chi_{\alpha}^{-1}$$ and $$\chi_{\beta}^{-1}$$ are also smooth. Therefore, the composite map $$\chi_{\beta} \circ \chi_{\alpha}^{-1}$$ is a smooth function.

Finally, we need to check the third condition of the definition of a manifold. For any two distinct points $$x \in U_{\alpha}$$ and $$\tilde{x} \in U_{\beta}$$, we need to find open subsets $$W \subset V_{\alpha}$$ and $$\tilde{W} \subset V_{\beta}$$ such that $$\chi_{\alpha}(x) \in W$$, $$\chi_{\beta}(\tilde{x}) \in \tilde{W}$$, and $$\chi_{\alpha}^{-1}(W) \cap \chi_{\beta}^{-1}(\tilde{W}) = \emptyset$$. This condition is satisfied because for any two distinct points on the sphere, we can always find small enough neighborhoods around these points that do not intersect.

Therefore, $$S^2$$ is a 2-dimensional manifold.

In code,

```python
# Define the number of points to generate
num_points = 1000

# Generate random spherical coordinates
theta = 2 * torch.pi * torch.rand(num_points)
phi = torch.acos(2 * torch.rand(num_points) - 1)

# Convert spherical coordinates to Cartesian coordinates
x = torch.sin(phi) * torch.cos(theta)
y = torch.sin(phi) * torch.sin(theta)
z = torch.cos(phi)

# Convert Cartesian coordinates to parameters of the stereographic projection
u = x / (1 - z)
v = y / (1 - z)

# Convert parameters of the stereographic projection to Cartesian coordinates
denominator = 1 + u**2 + v**2
x_prime = 2 * u / denominator
y_prime = 2 * v / denominator
z_prime = (-1 + u**2 + v**2) / denominator

# Convert Cartesian coordinates to spherical coordinates
theta_prime = torch.atan2(y_prime, x_prime)
phi_prime = torch.acos(z_prime)

# Adjust the range of theta_prime to [0, 2*pi]
theta_prime = (theta_prime + 2 * torch.pi) % (2 * torch.pi)

# Check that the original and final spherical coordinates are the same
print(torch.allclose(theta, theta_prime, atol=1e-6))
print(torch.allclose(phi, phi_prime, atol=1e-6))
```

```bash
True
```

## Review of the Verification of the Manifold Property of $$S^2$$

In this notebook, we have computationally verified that the 2-dimensional sphere $$S^2$$ is indeed a 2-dimensional manifold. We have done this by demonstrating that two different parameterizations of $$S^2$$ (spherical coordinates and stereographic projection) are equivalent and cover the same set $$S^2$$.

Specifically, we have:

1. Generated random points on $$S^2$$ using spherical coordinates.
2. Transformed these points to the parameters of the stereographic projection.
3. Transformed these parameters back to spherical coordinates.

The fact that the original and final spherical coordinates are the same (to within a specified tolerance) confirms that the two parameterizations are equivalent and cover the same set $$S^2$$.

This result is significant because it demonstrates that different choices of local coordinate charts can satisfy the definition of a manifold. This is a key property of manifolds and is fundamental to their study in differential geometry and related fields.

## Next Steps: Exploring the Torus

Having explored the manifold properties of the sphere $$S^2$$, we will next turn our attention to another important 2-dimensional manifold: the torus. The torus can be thought of as the Cartesian product of the circle $$S^1$$ with itself. In the following sections, we will explore the properties of the torus and demonstrate its manifold structure.

## The Torus as a 2-Dimensional Manifold

The torus, often visualized as the shape of a doughnut or an inner tube, is another example of a 2-dimensional manifold. It can be thought of as the Cartesian product of the circle $$S^1$$ with itself, denoted as $$S^1 \times S^1$$.

We can parameterize the torus using two angles, $$\theta$$ and $$\phi$$, which correspond to rotations around the two circular directions of the torus. Given a major radius $$R$$ and a minor radius $$r$$, the parameterization in Cartesian coordinates is given by:

$$
\begin{align*}
x &= (R + r\cos\theta)\cos\phi \\
y &= (R + r\cos\theta)\sin\phi \\
z &= r\sin\theta
\end{align*}
$$

where $$\theta, \phi \in [0, 2\pi)$$. This parameterization covers the entire torus except for a single point, which can be covered by a second parameterization.

Let's generate and plot points on the torus using this parameterization:

```python
# Define the major and minor radii
R = 30
r = 13

# Define the number of points to generate
num_points = 100000  # Increase the number of points

# Generate random angles theta and phi
theta = 2 * torch.pi * torch.rand(num_points)
phi = 2 * torch.pi * torch.rand(num_points)

# Calculate the Cartesian coordinates
x = (R + r * torch.cos(theta)) * torch.cos(phi)
y = (R + r * torch.cos(theta)) * torch.sin(phi)
z = r * torch.sin(theta)

# Plot the points on the torus
fig = plt.figure(figsize=(10, 10))  # Increase the size of the figure
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x.numpy(), y.numpy(), z.numpy(), alpha=0.6, edgecolors='w', s=20)
ax.set_box_aspect([1,1,1])  # Make the aspect ratio equal
plt.show()
```

<div class="imgcap_noborder">
<img src="/assets/torus.png">
</div>

# Lie Groups

A Lie group is a group that is also a differentiable manifold, such that the group operations (multiplication and inversion) are smooth. This means that a Lie group is a set that is equipped with a group structure, a manifold structure, and these structures are compatible in the sense that group operations are smooth functions.

Let's break down the definition:

1. **Group Structure:** A group is a set $$G$$ equipped with an operation $$\cdot: G \times G \rightarrow G$$ (often written multiplicatively) and an inversion operation $$^{-1}: G \rightarrow G$$ such that the following axioms are satisfied:

   - **Closure:** For all $$a, b \in G$$, the result of the operation $$a \cdot b$$ is also in $$G$$.
   - **Associativity:** For all $$a, b, c \in G$$, the equation $$(a \cdot b) \cdot c = a \cdot (b \cdot c)$$ holds.
   - **Identity element:** There is an element $$e \in G$$ such that for every element $$a \in G$$, the equations $$e \cdot a = a$$ and $$a \cdot e = a$$ hold.
   - **Inverse element:** For each element $$a \in G$$, there exists an element $$b \in G$$ such that $$a \cdot b = e$$ and $$b \cdot a = e$$.

2. **Manifold Structure:** As we discussed earlier, a manifold is a topological space that locally resembles Euclidean space. In the case of a Lie group, we require the manifold to be differentiable, meaning that we can do calculus on it. 

3. **Compatibility of Structures:** The group operations (multiplication and inversion) are required to be smooth functions when considered as maps between manifolds. More formally, if we denote the multiplication operation by $$\mu: G \times G \rightarrow G$$ (so that $$\mu(g, h) = g \cdot h$$) and the inversion operation by $$i: G \rightarrow G$$ (so that $$i(g) = g^{-1}$$), then $$\mu$$ and $$i$$ are required to be smooth.

An example of a Lie group is the general linear group $$GL(n, \R)$$, which consists of all $$n \times n$$ invertible matrices with real entries. The group operation is matrix multiplication, and the manifold structure comes from identifying each matrix with a point in $$\R^{n^2}$$. The group operations are smooth functions, so $$GL(n, \R)$$ is a Lie group.

Another example is the circle $$S^1$$ with the operation of complex multiplication. Each point on the circle can be identified with a complex number of absolute value 1, and multiplication of such numbers is a smooth operation.

Let's consider the general linear group $$GL(2, \R)$$ for simplicity. This group consists of all $$2 \times 2$$ invertible matrices with real entries. A general element of $$GL(2, \R)$$ can be written as:

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix}
$$

where $$a$$, $$b$$, $$c$$, and $$d$$ are real numbers and $$ad - bc \neq 0$$ (the condition for the matrix to be invertible).

The group operation is matrix multiplication, and the inverse of a matrix is given by:

$$
A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}
$$

Now, let's consider some subgroups of $$GL(2, \R)$$:

1. **Orthogonal Group $$O(2)$$:** This is the group of $$2 \times 2$$ matrices that preserve the Euclidean norm, i.e., $$AA^T = A^TA = I$$. The determinant of such matrices is either 1 or -1. A general element of $$O(2)$$ can be written as:

    $$
    O = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix} \quad \text{or} \quad \begin{bmatrix} \cos \theta & \sin \theta \\ \sin \theta & -\cos \theta \end{bmatrix}
    $$

    where $$\theta$$ is a real number.

2. **Special Orthogonal Group $$SO(2)$$:** This is the subgroup of $$O(2)$$ consisting of matrices with determinant 1. These are rotations in the plane. A general element of $$SO(2)$$ can be written as:

    $$
    SO = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix}
    $$

    where $$\theta$$ is a real number.

These subgroups are also Lie groups, as they are groups and differentiable manifolds, and the group operations are smooth. They are also examples of compact Lie groups, as they are closed and bounded subsets of $$\R^{2 \times 2}$$.

## Group Operations

In the case of $$GL(2, \R)$$, the group operation is matrix multiplication.

Matrix multiplication is a binary operation that takes a pair of matrices, and produces another matrix. For $$2 \times 2$$ matrices, the multiplication is defined as follows:

If we have two matrices $$A$$ and $$B$$ in $$GL(2, \R)$$, where

$$
A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \quad \text{and} \quad B = \begin{bmatrix} e & f \\ g & h \end{bmatrix}
$$

their product $$AB$$ is given by

$$
AB = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} e & f \\ g & h \end{bmatrix} = \begin{bmatrix} ae + bg & af + bh \\ ce + dg & cf + dh \end{bmatrix}
$$

This operation is associative, meaning that for any three matrices $$A\$$, $$B$$, and $$C$$ in $$GL(2, \R)$$, we have $$(AB)C = A(BC)$$.

**Example:**

Let's consider two specific matrices in $$GL(2, \R)$$:

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \quad \text{and} \quad B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
$$

Their product is given by

$$
AB = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1*5 + 2*7 & 1*6 + 2*8 \\ 3*5 + 4*7 & 3*6 + 4*8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$

So, the product of $$A$$ and $$B$$ is,
$$
\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
$$
demonstrating the closure property of the group.


Below, we show that the exponentiated value of simple 2x2 generator matrices is equal to the group of rotation matrices, a simple result with extremely significant implications. 

```python
# Define a function to generate a skew-symmetric matrix
def skew_symmetric(theta):
    return theta * torch.tensor(
        [[0, -1], 
         [1, 0]]
        )

# Define a vector
v = torch.tensor([1.0, 0.0])

# Generate a sequence of skew-symmetric matrices and compute their matrix exponentials
thetas = torch.linspace(0, 0.1, 10)
skew_symmetric_matrices = [skew_symmetric(theta) for theta in thetas]
rotation_matrices = [torch.linalg.matrix_exp(X) for X in skew_symmetric_matrices]

# Apply the rotation matrices to the vector
v_rotated = [R @ v for R in rotation_matrices]

# Plot the original and rotated vectors
plt.figure(figsize=(6,6))
plt.quiver(*v, angles='xy', scale_units='xy', scale=1, color='r')
for v_r in v_rotated:
    plt.quiver(*v_r, angles='xy', scale_units='xy', scale=1, color='b', alpha=0.2)
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.grid()
plt.show()
```

<div class="imgcap_noborder">
<img src="/assets/rotation.png" width="400">
</div>

In this way, we can see that the symmetry groups of transformations of objects in 2D space can be represented by the group of rotation matrices, which can be generated by 2x2 real matrices.

## Exponentiating a Matrix: Generating a Lie Group

In the previous section, we saw that the group of rotation matrices $$SO(2)$$ can be generated by a set of 2x2 real matrices. In this section, we will see how to generate a Lie group from a set of matrices. The elements of this set are called **generators** of the Lie group, and they themselves (along with the transformations they generate) are called a **Lie algebra** (more specifically, they are said to *satify the Lie algebra*).

Consider the simplest case of a 2x2 generator matrix, also known as a skew-symmetric matrix:

$$
G = \begin{bmatrix} 0 & -\theta \\ \theta & 0 \end{bmatrix}
$$

where $$\theta$$ is a scalar. You can observe that this matrix is skew-symmetric, i.e., $$G^T = -G$$.

Now, let's exponentiate this matrix $$G$$ using the matrix exponential function $$\exp$$. The matrix exponential is a power series defined as:

$$
\exp(G) = I + G + \frac{1}{2!} G^2 + \frac{1}{3!} G^3 + \dots = \sum_{k=0}^{\infty} \frac{1}{k!} G^k
$$

We can compute the first few powers of $$G$$:

$$
G^0 = I, \quad G^1 = G, \quad G^2 = \begin{bmatrix} -\theta^2 & 0 \\ 0 & -\theta^2 \end{bmatrix}, \quad G^3 = -\theta G, \quad G^4 = \theta^2 I, \quad \dots
$$

Now, we plug these matrix powers into the power series and separate the even and odd terms:

$$
\exp(G) = (I + \frac{1}{2!} G^2 + \frac{1}{4!} G^4 + \dots) + (G + \frac{1}{3!} G^3 + \dots) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
$$

As a result, the matrix exponential of the 2x2 skew-symmetric matrix generates the special orthogonal group $$SO(2)$$, which is the group of rotation matrices.

These results have significant implications for understanding how to apply Lie groups and matrix exponential to deep learning models, such as Transformer-based architectures. By leveraging the properties of exponentiated generator matrices and understanding the underlying structure, researchers can design models that are more robust and efficient when handling different types of data. Moreover, the idea of matrix exponentiation facilitates a natural way to interpolate between different network parameters when considering weight sharing, encouraging smooth behavior.

### Lie Algebra

The set of generators of a Lie group is called a **Lie algebra**. The Lie algebra is a vector space, and the generators are its basis vectors. The Lie algebra is closed under the commutator operation, which is defined as:

$$
[A, B] = AB - BA
$$

where $$A$$ and $$B$$ are elements of the Lie algebra. The commutator operation is also known as the **Lie bracket**. The Lie bracket is a bilinear operation, which means that it is linear in both arguments. The Lie bracket is also antisymmetric, which means that $$[A, B] = -[B, A]$$.

The Lie algebra is also closed under the scalar multiplication operation, which is defined as:

$$
\alpha A = A \alpha
$$

where $$\alpha$$ is a scalar and $$A$$ is an element of the Lie algebra.

The Lie algebra is also closed under the Jacobi identity, which is defined as:

$$
[A, [B, C]] + [B, [C, A]] + [C, [A, B]] = 0
$$

where $$A$$, $$B$$, and $$C$$ are elements of the Lie algebra.

The Lie algebra is also closed under the adjoint operation, which is defined as:

$$
\text{ad}(A) B = [A, B]
$$

There are additional properties of the Lie algebra, but these are the most important ones for our purposes of introducing the concepts. But for those curious, I would encourage you to look up Clifford algebras...

=========

### How does this relate to neural networks?

Consider a sequence of input data $$x_1, x_2, \dots, x_n$$. These data points can be visualized in a high-dimensional space. One of the main components of the Transformer architecture is the self-attention mechanism, which computes an attention score for each element within a sequence. The attention mechanism represents relations between elements in the sequence geometrically, using dot products between those elements in the high-dimensional space. 

By applying continuous transformations to this high-dimensional space, one could potentially extract additional information about the structures embedded in the input data. Lie groups play an important role in this regard. A continuous transformation in a high-dimensional space can be represented as an action of a Lie group on the manifold of data points. In practice, elements of a Lie group are given by the exponentiation of Lie algebra elements, which are closely related to matrix exponentials.

Suppose we have a Lie group in the form of a matrix exponential, as shown before:

$$
\exp(G) = \begin{bmatrix} \cos(\theta) & -\sin(\theta) \\ \sin(\theta) & \cos(\theta) \end{bmatrix}
$$

Applying this transformation to the input data in the high-dimensional space would result in a new representation of the data points. The transformed data points can be further used as input to a Transformer layer. This transformed representation might allow the attention mechanism to focus on different aspects of the input data and can potentially capture more complex relational structures present.

However, this approach has not yet been fully explored in the Transformer architectures, and most research has focused on finding more efficient ways to apply the attention mechanism, rather than incorporating geometric transformations explicitly. One potential direction for future research could be to consider the effect of applying transformations from special types of Lie groups on attention scores and relevance of input data points, and observe the impact this might have on model performance.

That being said, directly applying Lie group transformations as shown might not be the most natural or efficient way to incorporate the power of Lie groups and their symmetries into Transformer-based architectures. A more elegant approach would be to explore how Lie groups could be integrated into the design of Transformer networks inherently.

One possibility is to incorporate equivariance to Lie group actions into the self-attention mechanism. The principle of equivariance implies that the output of a function should transform in the same way as the input under a given transformation. In this context, it means that the attention mechanism should be designed such that it remains unchanged under the action of a Lie group transformation applied to input data.

To incorporate this idea into the self-attention mechanism, we need to rethink the computation of attention scores. Currently, attention scores are computed using a dot product between the query, key, and value vectors. Instead, we could design an attention mechanism that computes the scores after some consideration of the Lie group transformations.

For example, considering a Lie group of rotations, the design could compute attention scores in a rotation-invariant manner. This would involve redefining the computation of attention scores as the similarity between input element embeddings up to rotations (Lie group actions), rather than solely relying on dot products, which are not rotation-invariant
