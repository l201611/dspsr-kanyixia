/***************************************************************************
 *
 *   Copyright (C) 2004-2025 by Willem van Straten
 *   Licensed under the Academic Free License version 2.1
 *
 ***************************************************************************/

#include "dsp/ResponseProduct.h"

using namespace std;

unsigned dsp::ResponseProduct::get_npol() const
{
  if (!lazy_evaluation && has_changed())
    const_cast<ResponseProduct*>(this)->match(nullptr);
  return npol;
}

unsigned dsp::ResponseProduct::get_nchan() const
{
  if (!lazy_evaluation && has_changed())
    const_cast<ResponseProduct*>(this)->match(nullptr);
  return nchan;
}

unsigned dsp::ResponseProduct::get_ndat() const
{
  if (!lazy_evaluation && has_changed())
    const_cast<ResponseProduct*>(this)->match(nullptr);
  return ndat;
}

unsigned dsp::ResponseProduct::get_ndim() const
{
  if (!lazy_evaluation && has_changed())
    const_cast<ResponseProduct*>(this)->match(nullptr);
  return ndim;
}

//! Add a response to the product
void dsp::ResponseProduct::add_response (Response* _response)
{
  for (auto resp: multiplicands)
    if (resp.get() == _response)
      throw Error (InvalidParam, "dsp::ResponseProduct::add_response",
                    "response already added to multiplicands");

  multiplicands.push_back (_response);

  if (!has_optimal_fft() && _response->has_optimal_fft())
  {
    // adopt the optimal FFT length policy of the first multiplicand to have one
    set_optimal_fft(_response->get_optimal_fft());
  }

  this->changed();

  if (verbose)
    cerr << "dsp::ResponseProduct::add_response ptr=" << _response << " new size=" << multiplicands.size() << endl;
}

void dsp::ResponseProduct::configure (const Observation* obs, unsigned nchan)
{
  if (verbose)
    cerr << "dsp::ResponseProduct::configure size=" << multiplicands.size() << " nchan=" << nchan << endl;

  impulse_neg = 0;
  impulse_pos = 0;

  if (obs)
  {
    lazy_evaluation = true;
  }
  
  bool expand_ndat_as_required = false;
  if (ndat == 0)
  {
    if (verbose)
      cerr << "dsp::ResponseProduct::configure will expand ndat as required" << endl;
    expand_ndat_as_required = true;
  }

  for (unsigned iresp=0; iresp < multiplicands.size(); iresp++)
  {
    /*
      Fix for bugs/128 Ensure that the downsampling flag of each
      multiplicand matches that of the product.
    */
    multiplicands[iresp]->set_downsampling(downsampling);

    if (obs)
      multiplicands[iresp]->configure (obs, nchan);

    if (multiplicands[iresp]->has_changed())
      this->changed();

    impulse_neg += multiplicands[iresp]->impulse_neg;
    impulse_pos += multiplicands[iresp]->impulse_pos;

    if (verbose)
      cerr << "dsp::ResponseProduct::configure index=" << iresp
           << " impulse_neg=" << impulse_neg << " impulse_pos=" << impulse_pos << endl;

    ndim = std::max (ndim, multiplicands[iresp]->get_ndim());

    /*
      Fix for bugs/127 If the frequency resolution of any multiplicand is set,
      then it determines the frequency resolution of the product.
    */
    auto set_freq_res = multiplicands[iresp]->user_set_frequency_resolution;

    if (set_freq_res)
    {
      if (user_set_frequency_resolution == 0)
      {
        if (verbose)
          cerr << "dsp::ResponseProduct::configure setting frequency resolution "
                  "to that of multiplicand[" << iresp << "]=" << set_freq_res << endl;
        ndat = user_set_frequency_resolution = set_freq_res;
      }
      else if (user_set_frequency_resolution != set_freq_res)
      {
        // If already set, and not equal, then there is a conflict/error
        throw Error (InvalidState, "dsp::ResponseProduct::configure",
                     "set frequency resolution of multiplicand[%u]=%u does not equal frequency resolution of product=%u",
                    iresp, set_freq_res, user_set_frequency_resolution);
      }
    }

    if (expand_ndat_as_required && user_set_frequency_resolution == 0)
    {
      if (verbose)
        cerr << "dsp::ResponseProduct::configure index=" << iresp << " ndat=" << multiplicands[iresp]->get_ndat() << endl;
      ndat = std::max (ndat, multiplicands[iresp]->get_ndat());
    }
  }

  if (verbose)
    cerr << "dsp::ResponseProduct::configure ndat=" << ndat << endl;
}

void dsp::ResponseProduct::build (const Observation* input)
{
  if (verbose)
    cerr << "dsp::ResponseProduct::build size=" << multiplicands.size()
         << " nchan=" << nchan << " ndat=" << ndat
         << " impulse_pos=" << impulse_pos
         << " impulse_neg=" << impulse_neg << endl;

  if (multiplicands.size() == 0)
  {
    throw Error (InvalidState, "dsp::ResponseProduct::build", "no responses in product");
  }

  try
  {
    unsigned copy_ndim = 0;
    unsigned copy_index = 0;

    for (unsigned iresp=0; iresp < multiplicands.size(); iresp++)
    {
      if (verbose)
        cerr  << "dsp::ResponseProduct::build match_shape index=" << iresp << endl;

      multiplicands[iresp]->match_shape (this);

      if (multiplicands[iresp]->built == false)
      {
        if (verbose)
          cerr  << "dsp::ResponseProduct::build build index=" << iresp << endl;

        /*
          ResponseProduct is declared as a friend of Response so that it can set the built attribute, 
          which is normally set only in Response::rebuild.  We don't want to call Response::rebuild
          here because this method also calls swap_as_needed, and swap_as_needed should be called
          only once, on the result of this ResponseProduct.  This is also why naturalize is called
          below; some Response objects may also directly build a (partially) swapped response.
        */
        multiplicands[iresp]->verify_dataspace();
        multiplicands[iresp]->build(input);
        multiplicands[iresp]->built = true;

        // before computing products of multiple response functions, ensure that they are ordered identically
        multiplicands[iresp]->naturalize();
      }

      unsigned resp_ndim = multiplicands[iresp]->get_ndim();

      if (resp_ndim > copy_ndim)
      {
        if (verbose)
        {
          cerr << "dsp::ResponseProduct::build multiplicands[" << iresp << "]->ndim=" << resp_ndim
               << " > copy_ndim=" << copy_ndim << endl
               << "dsp::ResponseProduct::build changing copy_index from " << copy_index << " to " << iresp << endl;
        }
        copy_index = iresp;
        copy_ndim = resp_ndim;
      }
    }

    if (verbose)
      cerr << "dsp::ResponseProduct::build copy from index=" << copy_index << endl;

    /*
     * ResponseProduct::configure correctly sets Response::impulse_pos and Response::impulse_neg
     * to the sum of the impulse_pos and impulse_neg attributes of each of the multiplicands in
     * the product.  The following call to Response::copy will incorrectly set impulse_pos and
     * impulse_neg equal to the attributes of the single multiplicand indexed by copy_index.
     * Therefore, backup the correct values and restore them after calling Response::copy.
     * ResponseProductTest test_impulse_pos_and_neg verifys this correct behaviour.
     */

    unsigned backup_impulse_pos = impulse_pos;
    unsigned backup_impulse_neg = impulse_neg;

    copy (multiplicands[copy_index]);

    impulse_pos = backup_impulse_pos;
    impulse_neg = backup_impulse_neg;

    if (verbose)
      cerr << "dsp::ResponseProduct::build ndat=" << ndat << " nchan=" << nchan << endl;

    for (unsigned iresp=0; iresp < multiplicands.size(); iresp++)
    {
      if (iresp != copy_index)
      {
        if (verbose)
          cerr << "dsp::ResponseProduct::build multiply by index=" << iresp << endl;
        multiply (multiplicands[iresp]);
      }
    }
  }
  catch (Error& error)
  {
    throw error += "dsp::ResponseProduct::build";
  }

  if (verbose)
  {
    cerr << "dsp::ResponseProduct::build this=" << (void*) this
      << " ndat=" << ndat
      << " impulse_pos=" << impulse_pos
      << " impulse_neg=" << impulse_neg
      << std::endl;
  }
}

void dsp::append_response (Reference::To<Response>& composite, Response* component)
{
  if (!composite)
  {
    composite = component;
    return;
  }

  auto product = dynamic_cast<ResponseProduct*> (composite.get());

  if (!product)
  {
    product = new ResponseProduct;
    product->add_response(composite);
    composite = product;
  }

  product->add_response(component);
}
