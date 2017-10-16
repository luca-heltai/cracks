#ifndef BITMAP_FILE_H
#define BITMAP_FILE_H

#include <deal.II/base/config.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>

#include <vector>
#include <fstream>

using namespace dealii;

// For Example 3 (multiple cracks in a heterogenous medium)
// reads .pgm file and returns it as floating point values
// taken from step-42
class BitmapFile
{
public:
  BitmapFile(const std::string &name);

  double
  get_value(const double x, const double y) const;

private:
  std::vector<double> image_data;
  double hx, hy;
  int nx, ny;

  double
  get_pixel_value(const int i, const int j) const;
};


template <int dim>
class BitmapFunction : public Function<dim>
{
public:
  BitmapFunction(const std::string &filename,
                 double x1_, double x2_, double y1_, double y2_, double minvalue_, double maxvalue_)
    : Function<dim>(1),
      f(filename), x1(x1_), x2(x2_), y1(y1_), y2(y2_), minvalue(minvalue_), maxvalue(maxvalue_)
  {}

  virtual
  double value (const Point<dim> &p,
                const unsigned int /*component*/) const
  {
    Assert(dim==2, ExcNotImplemented());
    double x = (p(0)-x1)/(x2-x1);
    double y = (p(1)-y1)/(y2-y1);
    return minvalue + f.get_value(x,y)*(maxvalue-minvalue);
  }
private:
  BitmapFile f;
  double x1,x2,y1,y2;
  double minvalue, maxvalue;
};



#endif // BITMAP_FILE_H
