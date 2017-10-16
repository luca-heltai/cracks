#ifndef TENSORS_H
#define TENSORS_H


// Define some tensors for cleaner notation later.
namespace Tensors
{

  template <int dim>
  inline Tensor<1, dim>
  get_grad_pf (
    unsigned int q,
    const std::vector<std::vector<Tensor<1, dim> > > &old_solution_grads)
  {
    Tensor<1, dim> grad_pf;
    grad_pf[0] = old_solution_grads[q][dim][0];
    grad_pf[1] = old_solution_grads[q][dim][1];

    return grad_pf;
  }

  template <int dim>
  inline Tensor<2, dim>
  get_grad_u (
    unsigned int q,
    const std::vector<std::vector<Tensor<1, dim> > > &old_solution_grads)
  {
    Tensor<2, dim> structure_continuation;
    structure_continuation[0][0] = old_solution_grads[q][0][0];
    structure_continuation[0][1] = old_solution_grads[q][0][1];
    structure_continuation[1][0] = old_solution_grads[q][1][0];
    structure_continuation[1][1] = old_solution_grads[q][1][1];

    return structure_continuation;
  }

  template <int dim>
  inline Tensor<2, dim>
  get_Identity ()
  {
    Tensor<2, dim> identity;
    identity[0][0] = 1.0;
    identity[0][1] = 0.0;
    identity[1][0] = 0.0;
    identity[1][1] = 1.0;

    return identity;
  }

  template <int dim>
  inline Tensor<1, dim>
  get_u (
    unsigned int q,
    const std::vector<Vector<double> > &old_solution_values)
  {
    Tensor<1, dim> u;
    u[0] = old_solution_values[q](0);
    u[1] = old_solution_values[q](1);

    return u;
  }

  template <int dim>
  inline Tensor<1, dim>
  get_u_LinU (
    const Tensor<1, dim> &phi_i_u)
  {
    Tensor<1, dim> tmp;
    tmp[0] = phi_i_u[0];
    tmp[1] = phi_i_u[1];

    return tmp;
  }

}


#endif // TENSORS_H
