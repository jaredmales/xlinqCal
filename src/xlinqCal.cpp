/** \file xlinqCal.cpp
 * \brief The xlinqCal application entry point.
 *
 * \ingroup xlinqCal_files
 */

#include "xlinqCal.hpp"

/// Run the xlinqCal application.
int main( int    argc, /**< [in] the number of command line arguments */
          char **argv /**< [in] the command line arguments */ )
{
    xlinqCal::xlinqCal app;

    return app.main( argc, argv );
}
