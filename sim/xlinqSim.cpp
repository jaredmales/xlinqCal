/** \file xlinqSim.cpp
 * \brief Standalone simulator for exercising xlinqCal on a workstation.
 *
 * \ingroup xlinqCal_files
 */

#include <csignal>
#include <iostream>

#include <mx/app/application.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>

#include "xlinqSim.hpp"

/// Set by the signal handler to stop the simulator.
static volatile std::sig_atomic_t g_stop = 0;

/// Signal handler, requests a stop.
static void handleSignal( int sig /**< [in] the signal number */ )
{
    static_cast<void>( sig );
    g_stop = 1;
}

/// Command line front end for xlinqSim.
/** Runs the simulator until interrupted. On startup it writes the response matrices and a random
 * poke cube to FITS files with the stream prefix, so xlinqCal results can be checked against them.
 */
class xlinqSimApp : public mx::app::application
{
  protected:
    xlinqCal::xlinqSimConfig m_cfg; ///< The simulator configuration.

    int m_nPokes{ 10 }; ///< Number of random pokes written to `<prefix>pokes.fits`.

    int m_nModes{ 0 }; ///< Number of simulated modal coefficients in `<prefix>modevalWFS`, 0 disables.

  public:
    /// Set up the configuration targets.
    virtual void setupConfig();

    /// Load the configuration.
    virtual void loadConfig();

    /// Run the simulator until interrupted.
    virtual int execute();
};

inline void xlinqSimApp::setupConfig()
{
    using mx::app::argType;

    config.add( "sim.prefix",
                "",
                "sim.prefix",
                argType::Required,
                "sim",
                "prefix",
                false,
                "string",
                "Prefix for all stream and file names. Default xlsim_." );
    config.add( "sim.wfsFps",
                "",
                "sim.wfsFps",
                argType::Required,
                "sim",
                "wfsFps",
                false,
                "float",
                "camwfs frame rate. Default 2000." );
    config.add( "sim.sciFps",
                "",
                "sim.sciFps",
                argType::Required,
                "sim",
                "sciFps",
                false,
                "float",
                "camsci1 frame rate. Default 4." );
    config.add( "sim.sciGain",
                "",
                "sim.sciGain",
                argType::Required,
                "sim",
                "sciGain",
                false,
                "float",
                "camsci1 rms response to 1 um rms [counts]. Default 50." );
    config.add( "sim.wfsGain",
                "",
                "sim.wfsGain",
                argType::Required,
                "sim",
                "wfsGain",
                false,
                "float",
                "camwfs rms response to 1 um rms [counts]. Default 50." );
    config.add(
        "sim.nModes",
        "",
        "sim.nModes",
        argType::Required,
        "sim",
        "nModes",
        false,
        "int",
        "Number of modal coefficients in <prefix>modevalWFS (like aol1_modevalWFSpub), 0 disables. Default 0." );
    config.add( "sim.sciQuadGain",
                "",
                "sim.sciQuadGain",
                argType::Required,
                "sim",
                "sciQuadGain",
                false,
                "float",
                "camsci1 rms |H c| for 1 um rms; adds |H c|^2 [counts]. Default 0 (off)." );
    config.add( "sim.drift",
                "",
                "sim.drift",
                argType::Required,
                "sim",
                "drift",
                false,
                "float",
                "Fractional level drift per second. Default 0.001." );
    config.add( "sim.nPokes",
                "",
                "sim.nPokes",
                argType::Required,
                "sim",
                "nPokes",
                false,
                "int",
                "Number of random pokes written to <prefix>pokes.fits. Default 10." );
}

inline void xlinqSimApp::loadConfig()
{
    config( m_cfg.prefix, "sim.prefix" );
    config( m_cfg.wfsFps, "sim.wfsFps" );
    config( m_cfg.sciFps, "sim.sciFps" );
    config( m_cfg.sciGain, "sim.sciGain" );
    config( m_cfg.wfsGain, "sim.wfsGain" );
    config( m_nModes, "sim.nModes" );
    config( m_cfg.sciQuadGain, "sim.sciQuadGain" );
    config( m_cfg.drift, "sim.drift" );
    config( m_nPokes, "sim.nPokes" );

    m_cfg.dmChannel = m_cfg.prefix + "dm01disp04";
    m_cfg.dmComb    = m_cfg.prefix + "dm01disp";
    m_cfg.wfsName   = m_cfg.prefix + "camwfs";
    m_cfg.sciName   = m_cfg.prefix + "camsci1";

    if( m_nModes > 0 )
    {
        m_cfg.nModes      = m_nModes;
        m_cfg.modevalName = m_cfg.prefix + "modevalWFS";
    }
}

inline int xlinqSimApp::execute()
{
    signal( SIGINT, handleSignal );
    signal( SIGTERM, handleSignal );

    try
    {
        xlinqCal::xlinqSim sim( m_cfg );

        mx::fits::fitsFile<float> ff;
        ff.write( m_cfg.prefix + "wfsR.fits", sim.wfsR() );
        ff.write( m_cfg.prefix + "sciR.fits", sim.sciR() );

        if( m_nModes > 0 )
        {
            ff.write( m_cfg.prefix + "modevalR.fits", sim.modevalR() );
        }

        if( m_cfg.sciQuadGain > 0 )
        {
            ff.write( m_cfg.prefix + "sciHr.fits", sim.sciHr() );
            ff.write( m_cfg.prefix + "sciHi.fits", sim.sciHi() );
        }
        ff.write( m_cfg.prefix + "pokes.fits",
                  xlinqCal::xlinqSim::randomPokes( m_cfg.dmRows, m_cfg.dmCols, m_nPokes, m_cfg.seed + 10 ) );

        sim.start();

        std::cout << "simulating " << m_cfg.dmChannel << ", " << m_cfg.dmComb << ", " << m_cfg.wfsName << ", "
                  << m_cfg.sciName << ( m_nModes > 0 ? ", " + m_cfg.modevalName : std::string() )
                  << "; ctrl-c to stop\n";

        while( !g_stop )
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 100 ) );
        }

        sim.stop();
    }
    catch( const std::exception &e )
    {
        std::cerr << "xlinqSim: " << e.what() << "\n";
        return -1;
    }

    return 0;
}

/// Run the simulator.
int main( int    argc, /**< [in] the number of command line arguments */
          char **argv /**< [in] the command line arguments */ )
{
    xlinqSimApp app;

    return app.main( argc, argv );
}
