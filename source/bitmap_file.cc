#include "bitmap_file.h"

// The constructor of this class reads in the data that describes
// the obstacle from the given file name.
BitmapFile::BitmapFile(const std::string &name)
  :
  image_data(0),
  hx(0),
  hy(0),
  nx(0),
  ny(0)
{
  std::ifstream f(name.c_str());
  AssertThrow (f, ExcMessage (std::string("Can't read from file <") +
                              name + ">!"));

  std::string temp;
  getline(f, temp);
  f >> temp;
  if (temp[0]=='#')
    getline(f, temp);

  f >> nx >> ny;

  AssertThrow(nx > 0 && ny > 0, ExcMessage("Invalid file format."));

  for (int k = 0; k < nx * ny; k++)
    {
      unsigned int val;
      f >> val;
      image_data.push_back(val / 255.0);
    }

  hx = 1.0 / (nx - 1);
  hy = 1.0 / (ny - 1);
}

// The following two functions return the value of a given pixel with
// coordinates $i,j$, which we identify with the values of a function
// defined at positions <code>i*hx, j*hy</code>, and at arbitrary
// coordinates $x,y$ where we do a bilinear interpolation between
// point values returned by the first of the two functions. In the
// second function, for each $x,y$, we first compute the (integer)
// location of the nearest pixel coordinate to the bottom left of
// $x,y$, and then compute the coordinates $\xi,\eta$ within this
// pixel. We truncate both kinds of variables from both below
// and above to avoid problems when evaluating the function outside
// of its defined range as may happen due to roundoff errors.
double
BitmapFile::get_pixel_value(const int i,
                            const int j) const
{
  assert(i >= 0 && i < nx);
  assert(j >= 0 && j < ny);
  return image_data[nx * (ny - 1 - j) + i];
}

double
BitmapFile::get_value(const double x,
                      const double y) const
{
  const int ix = std::min(std::max((int) (x / hx), 0), nx - 2);
  const int iy = std::min(std::max((int) (y / hy), 0), ny - 2);

  const double xi  = std::min(std::max((x-ix*hx)/hx, 1.), 0.);
  const double eta = std::min(std::max((y-iy*hy)/hy, 1.), 0.);

  return ((1-xi)*(1-eta)*get_pixel_value(ix,iy)
          +
          xi*(1-eta)*get_pixel_value(ix+1,iy)
          +
          (1-xi)*eta*get_pixel_value(ix,iy+1)
          +
          xi*eta*get_pixel_value(ix+1,iy+1));
}
