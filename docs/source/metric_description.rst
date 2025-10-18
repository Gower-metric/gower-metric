===================
Metric Description
===================

It is key to understand how something works when using it.  
This section describes how Gower's metric formula works, in detail.

----------------------
Gower characteristics
----------------------

In 1971, Gower introduced a general similarity coefficient that encompasses several existing measures as special cases, making it adaptable to various scenarios.

Two individuals, *i* and *j*, can be compared on a variable *k* and assigned a score :math:`s_{ijk}`.  
The similarity between *i* and *j* is calculated as the weighted average of these scores across all comparisons:

.. math::

   S_{ij} = \frac{\sum_{k=1}^{p} s_{ijk}\delta_{ijk}}{\sum_{k=1}^{p} \delta_{ijk}}

Let :math:`\delta_{ijk}` represent the possibility of making a comparison. Specifically, :math:`\delta_{ijk} = 1` when variable *k* can be compared for individuals *i* and *j*, meaning no missing values exist for both.

The Gower's distance can be calculated as :math:`d_{ij} = 1 - S_{ij}`,  
and the scores :math:`s_{ijk}` are defined as follows:

**Binary symmetric data:**

.. math::

   s_{ijk} = 
   \begin{cases}
   1 & \text{if } x_{ik} = x_{jk} \\
   0 & \text{otherwise}
   \end{cases}

**Binary asymmetric data:**

.. math::

   s_{ijk} = 
   \begin{cases}
   1 & \text{if } x_{ik} = x_{jk} = 1 \\
   0 & \text{otherwise}
   \end{cases}

**Ratio scale:**

.. math::

   s_{ijk} = 1 - \frac{|x_{ik} - x_{jk}|}{R_{k}}, \quad \text{where } R_k = \max(x_k) - \min(x_k)

**Categorical nominal:**

.. math::

   s_{ijk} =
   \begin{cases}
   1 & \text{if } x_{ik} = x_{jk} \\
   0 & \text{otherwise}
   \end{cases}

Additionally, Gower proposed the inclusion of **weights** in the similarity coefficient.

----------------------
Metric enhancements
----------------------

In 1999 and 2021, several improvements to the original Gower's metric were proposed.  
Below are listed implemented enhancements.

**Ordinal variables**

The basic implementation of Gower’s distance does not account for ordinal variables.  
To address this, we can use the solution proposed by Podani (1999):

.. math::

   s_{ijk} = 
   \begin{cases}
   1 & \text{if } r_{ik} = r_{jk} \\
   1 - \frac{r_{ik} - r_{jk} - \frac{T_{ik} - 1}{2} - \frac{T_{jk} - 1}{2}}{\max(r_k) - \min(r_k) - \frac{T_{\max,k} - 1}{2} - \frac{T_{\min,k} - 1}{2}} & \text{otherwise}
   \end{cases}

where:

- :math:`r_{ik}` – rank of attribute *k* at element *i*
- :math:`T_{ik}` – the cardinality of elements with equal rank score to element *i* at attribute *k*

**Example of calculating rankings and cardinalities:**

.. list-table::
   :header-rows: 1
   :widths: 20 10 10 10 10 10 10 10 10

   * - Variable's value
     - 1
     - 2
     - 1
     - 4
     - 1
     - 2
     - 2
     - 1
   * - Variable's rank
     - 2.5
     - 6
     - 2.5
     - 8
     - 2.5
     - 6
     - 6
     - 2.5
   * - T - rank's cardinality
     - 4
     - 3
     - 4
     - 1
     - 4
     - 3
     - 3
     - 4

**Ratio scale improvements**

**Outliers compensation**

**Problem:** Outliers in numerical variables affect their contribution to the overall dissimilarity.  
**Solution:** Replace :math:`R_k` with :math:`IQR_k`, which is the Inter-Quartile Range (:math:`P_{75\%} - P_{25\%}`), or even Inter-Decile.

.. math::

   s_{ijk} = 
   \begin{cases}
   1 - \frac{|x_{ik} - x_{jk}|}{IQR_k} & \text{if } |x_{ik} - x_{jk}| < IQR_k \\
   0 & \text{otherwise}
   \end{cases}

**Categorical variables dominance reduction**

**Problem:** The Gower distance tends to treat units with the same categorical values as closer,  
giving less importance to the distance on ratio-scaled variables.  

**Solution:** Discretize the ratio-scaled variables using a non-fixed discretization scheme,  
like *Kernel Density Estimation (KDE)*.

.. math::

   s_{ijk} =
   \begin{cases}
   1 & \text{if } |x_{ik} - x_{jk}| \leq h_k \\
   \frac{|x_{ik} - x_{jk}|}{g_k} & \text{if } h_k < |x_{ik} - x_{jk}| < g_k \\
   0 & \text{if } |x_{ik} - x_{jk}| \geq g_k
   \end{cases}

Where:

- :math:`g_k` – can be :math:`IQR_k` or the total range of values at position *k*
- :math:`h_k` – the KDE's bandwidth, estimated using one of:

  - **Silverman:** :math:`h_k = \frac{c}{\sqrt[5]{n}} \min(\mathrm{std}_k, \frac{IQR_k}{1.34})`
  - **Scott:** :math:`h_k = \frac{c}{\sqrt[5]{n}} \mathrm{std}_k`
  - **Sheather-Jones:** minimizes asymptotic MISE  
    :math:`MISE(h) = E \int (\hat{f}(x;h) - f(x))^2 dx`

Best-fitting bandwidth values can also be found using **Grid Search** or **Optuna**.

----------------------
Weights optimization
----------------------

**Cophenetic Correlation Coefficient (CPCC)**

**Goal:** Find optimal weights for Gower's distance metric to maximize the Cophenetic Correlation Coefficient (CPCC)  
of the resulting hierarchical clustering.

Implementation based on `scipy.cluster.hierarchy.cophenet <https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.cophenet.html>`_:

.. math::

   C = \frac{\sum_{i<j}(x(i,j)-\bar{x})(t(i,j)-\bar{t})}{\sqrt{(\sum_{i<j}(x(i,j)-\bar{x})^2)(\sum_{i<j}(t(i,j)-\bar{t})^2)}}

where:  

- :math:`x(i,j)` – Gower's distance between *i* and *j*, with global mean :math:`\bar{x}`
- :math:`t(i,j)` – cophenetic distance between *i* and *j*, with global mean :math:`\bar{t}`

Since CPCC is differentiable, it can be optimized using **L-BFGS-B**.

**Index of Agreement (IoA)**

Indicates how accurately a model fits the actual data.

.. math::

   IoA = 1 - \frac{\sum_{i=1}^n (O_i - P_i)^2}{\sum_{i=1}^n (|P_i - \bar{O}| + |O_i - \bar{O}|)^2}

where:

- :math:`P_i` – predicted value of *i* (Gower's distance)
- :math:`O_i` – observed value of *i* (cophenetic distance)
- :math:`\bar{O}` – mean of observed values
- :math:`n` – number of observations