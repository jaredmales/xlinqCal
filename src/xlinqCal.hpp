/** \file xlinqCal.hpp
 * \brief The xlinqCal application: measure synchronized camwfs and camsci1 responses to DM pokes.
 *
 * \ingroup xlinqCal_files
 */

#ifndef xlinqCal_hpp
#define xlinqCal_hpp

#include <array>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include <mx/app/application.hpp>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/improc/milkImage.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>

#include "isioStream.hpp"

#ifndef XLINQCAL_GIT_VERSION
    #define XLINQCAL_GIT_VERSION "unknown"
#endif

/** \defgroup xlinqCal xlinqCal
 * \brief Standalone MagAO-X calibration of synchronized camwfs and camsci1 poke responses.
 *
 * The DM is poked by writing `amp * shape` to a DM channel (default `dm01disp04`), and the
 * camwfs and camsci1 frames taken while the poke is on the DM are averaged. Frames are
 * synchronized to camsci1: after a poke, the first camsci1 frame and `sci.nSettle` more are
 * discarded, then camwfs frames are averaged between the posts that bound `sci.nFrames` averaged
 * camsci1 frames. The response for each camera is `(mean(+) - mean(-))/2`, averaged over
 * `seq.nRepeat` repeats with alternating sign order (`+-`, `-+`, ...).
 */

/** \defgroup xlinqCal_files xlinqCal Files
 * \ingroup xlinqCal
 */

namespace xlinqCal
{

/// Column indices of the per-acquisition timing table.
namespace timingCol
{
enum : int
{
    poke,         ///< Poke index in the input cube.
    repeat,       ///< Repeat index.
    sign,         ///< Poke sign, +1 or -1.
    dmCnt0,       ///< cnt0 of the combined DM stream when the poke was confirmed, NaN if no DM write.
    dmProj,       ///< Projection of the combined DM change onto the commanded change, NaN if no DM write.
    sciCnt0First, ///< cnt0 of the first averaged camsci1 frame.
    sciCnt0Last,  ///< cnt0 of the last averaged camsci1 frame.
    sciN,         ///< Number of camsci1 frames averaged.
    sciMissed,    ///< camsci1 frames skipped (cnt0 gaps) during averaging.
    wfsCnt0First, ///< cnt0 of the first averaged camwfs frame.
    wfsCnt0Last,  ///< cnt0 of the last averaged camwfs frame.
    wfsN,         ///< Number of camwfs frames averaged.
    wfsMissed,    ///< camwfs frames skipped (cnt0 gaps) while the gate was open.
    wfsTorn,      ///< camwfs frames dropped because they changed during the copy.
    tStart,       ///< Gate open time, Unix seconds.
    tEnd,         ///< Gate close time, Unix seconds.
    nCols         ///< The number of columns.
};
} // namespace timingCol

/// One row of the timing table.
typedef std::array<double, timingCol::nCols> timingRecord;

/// The xlinqCal application.
/** Standalone (no INDI). Streams are opened with ImageStreamIO. The operator is responsible for
 * system setup, including making camsci1 available on the host (e.g. via shmimTCP).
 */
class xlinqCal : public mx::app::application
{
  protected:
    /** \name Configuration - Data
     * @{
     */

    std::string m_dmChannel{ "dm01disp04" }; ///< The DM channel stream poked by this app.

    std::string m_dmComb{ "dm01disp" }; ///< The combined DM stream whose update confirms a poke.

    double m_dmWait{ 2.0 }; ///< Timeout in seconds for the combined DM stream to update after a poke.

    double m_dmProjTol{ 0.25 }; ///< Warn if the combined DM projection differs from 1 by more than this.

    std::string m_pokeFile; ///< FITS cube of DM shapes, rows x cols x Npoke, normalized to 1 micron rms.

    float m_pokeAmp{ 0 }; ///< The poke amplitude, microns rms. Required, must be nonzero.

    int m_pokeFirst{ 0 }; ///< First poke index in m_pokeFile to measure.

    int m_pokeLast{ -1 }; ///< Last poke index in m_pokeFile to measure, -1 for the last in the file.

    std::string m_sciName{ "camsci1" }; ///< The science camera stream, which sets synchronization.

    unsigned m_sciSettle{ 1 }; ///< camsci1 frames discarded after the first frame following a poke.

    unsigned m_sciFrames{ 10 }; ///< camsci1 frames averaged per poke sign.

    double m_sciWait{ 5.0 }; ///< Timeout in seconds for each camsci1 frame.

    std::string m_wfsName{ "camwfs" }; ///< The WFS camera stream, averaged over the camsci1 windows.

    unsigned m_nRepeat{ 2 }; ///< Number of +/- pairs per poke, with alternating order.

    std::string m_outDir{ "." }; ///< Output directory, created if needed.

    std::string m_outPrefix; ///< Output file prefix. Default is xlinqCal_<UTC timestamp>.

    double m_checkpointMin{ 10 }; ///< Minutes between checkpoint writes of completed planes, 0 disables.

    double m_preflightTime{ 2.0 }; ///< Seconds to measure camera frame rates before starting.

    bool m_yes{ false }; ///< Skip the operator confirmation prompt.

    ///@}

    /** \name Streams and Data - Data
     * @{
     */

    isioStream m_sci; ///< The camsci1 stream, with claimed semaphore.

    isioStream m_wfs; ///< The camwfs stream, with claimed semaphore, read by the WFS thread.

    isioStream m_comb; ///< The combined DM stream, with claimed semaphore.

    mx::improc::milkImage<float> m_dm; ///< The DM channel, written non-passively so dmcomb updates.

    mx::improc::eigenImage<float> m_dmInitial; ///< DM channel content at startup, restored on exit.

    mx::improc::eigenImage<float> m_dmLast; ///< The last command written to the DM channel.

    bool m_dmWritten{ false }; ///< True once the DM channel has been written, so it needs restoring.

    mx::improc::eigenCube<float> m_pokes; ///< The poke shapes, normalized to 1 micron rms.

    mx::improc::eigenCube<float> m_sciResp; ///< The camsci1 response cube, one plane per measured poke.

    mx::improc::eigenCube<float> m_wfsResp; ///< The camwfs response cube, one plane per measured poke.

    int m_nDone{ 0 }; ///< Number of completed response planes.

    std::vector<timingRecord> m_timing; ///< One timing row per acquisition.

    double m_sciFps{ 0 }; ///< camsci1 frame rate measured during preflight.

    double m_wfsFps{ 0 }; ///< camwfs frame rate measured during preflight.

    uint64_t m_sciLastCnt0{ 0 }; ///< cnt0 of the last camsci1 frame received.

    std::string m_dateBeg; ///< UTC start of measurement, ISO 8601.

    std::string m_dateEnd; ///< UTC end of measurement, ISO 8601.

    std::ofstream m_logFile; ///< Run log, opened in the output directory.

    ///@}

    /** \name WFS Thread - Data
     * State shared between the main thread and the WFS thread is protected by m_wfsMutex.
     * @{
     */

    std::thread m_wfsThread; ///< The thread that reads and accumulates camwfs frames.

    std::atomic<bool> m_wfsRun{ false }; ///< True while the WFS thread should run.

    std::mutex m_wfsMutex; ///< Protects the gate and accumulators below.

    bool m_wfsGate{ false }; ///< When true, camwfs frames are accumulated.

    bool m_wfsGateFirst{ false }; ///< True until the first frame after the gate opens, for gap counting.

    mx::improc::eigenImage<double> m_wfsSum; ///< Sum of camwfs frames while the gate is open.

    uint64_t m_wfsN{ 0 }; ///< Number of frames in m_wfsSum.

    uint64_t m_wfsMissed{ 0 }; ///< cnt0 gaps while the gate is open.

    uint64_t m_wfsTorn{ 0 }; ///< Frames dropped as torn while the gate is open.

    uint64_t m_wfsCnt0First{ 0 }; ///< cnt0 of the first accumulated frame.

    uint64_t m_wfsCnt0Last{ 0 }; ///< cnt0 of the last accumulated frame.

    uint64_t m_wfsLastSeen{ 0 }; ///< cnt0 of the last frame seen by the WFS thread.

    std::atomic<int> m_wfsErrno{ 0 }; ///< Nonzero if the WFS thread exited on an error.

    ///@}

    /// Set by the signal handler to request a stop.
    static inline volatile std::sig_atomic_t s_stop{ 0 };

  public:
    /// Default c'tor.
    xlinqCal() = default;

    /// D'tor, stops the WFS thread and restores the DM channel if still needed.
    ~xlinqCal() noexcept;

    /// Get the sign sequence for a poke.
    /** Repeats alternate `+-` and `-+`, so 2 repeats give `+--+`, which cancels linear drift.
     */
    static std::vector<int> signSequence( unsigned nRepeat /**< [in] number of +/- pairs */ );

    /// Count the acquisitions in a sign sequence that follow a DM change and so need settling.
    static unsigned settleCount( const std::vector<int> &signs /**< [in] the sign sequence */ );

    /// Format a timestamp as UTC ISO 8601 with milliseconds.
    static std::string isoTime( const timespec &ts /**< [in] the time to format */ );

    /// Get the current time as Unix seconds.
    static double nowSec();

    /// Request a stop, as from a signal.
    static void requestStop();

    /// Clear a stop request.
    static void clearStop();

    /// Get the number of completed response planes.
    int nDone() const;

    /// Get the timing table.
    const std::vector<timingRecord> &timing() const;

    /// Get the camsci1 response cube.
    const mx::improc::eigenCube<float> &sciResp() const;

    /// Get the camwfs response cube.
    const mx::improc::eigenCube<float> &wfsResp() const;

  protected:
    /// Set up the configuration targets.
    virtual void setupConfig();

    /// Load the configuration.
    virtual void loadConfig();

    /// Run the calibration.
    /** \returns 0 on success
     * \returns 1 if stopped by request (outputs contain completed planes)
     * \returns -1 on error (outputs contain completed planes)
     */
    virtual int execute();

    /// Print a timestamped message to stdout and the run log.
    void log( const std::string &msg /**< [in] the message */ );

    /// Print a timestamped error to stderr and the run log.
    void logError( const std::string &msg /**< [in] the message */ );

    /// Load and validate the poke cube.
    /** \returns 0 on success, -1 on error
     */
    int loadPokes();

    /// Open the streams, check sizes, and save the DM channel's initial content.
    /** \returns 0 on success, -1 on error
     */
    int openStreams();

    /// Measure frame rates, print the run summary, and get operator confirmation.
    /** \returns 0 to proceed, 1 if the operator declined, -1 on error
     */
    int preflight();

    /// Create the output directory, set the prefix, and open the run log.
    /** \returns 0 on success, -1 on error
     */
    int setupOutput();

    /// The WFS thread entry point.
    void wfsThreadExec();

    /// Start the WFS thread.
    void startWfsThread();

    /// Stop and join the WFS thread.
    void stopWfsThread();

    /// Measure all pokes, with checkpoints.
    /** \returns 0 on success, 1 if stopped, -1 on error
     */
    int run();

    /// Measure the response for one poke, over all signs and repeats.
    /** \returns 0 on success, 1 if stopped, -1 on error
     */
    int measurePoke( int k /**< [in] the poke index in m_pokes */ );

    /// Write a DM command and wait for the combined DM stream to update.
    /** \returns 0 on success, 1 if stopped, -1 on error
     */
    int applyCommand( const mx::improc::eigenImage<float> &cmd, /**< [in] the full channel command */
                      timingRecord                        &rec /**< [in.out] DM fields are set */ );

    /// Wait for the next new camsci1 frame, optionally copying it.
    /** \returns 0 on success, 1 if stopped, -1 on timeout or error
     */
    int waitSciFrame( mx::improc::eigenImage<double> *frame, /**< [out] the frame, or nullptr to only wait */
                      uint64_t                       &cnt0 /**< [out] the frame counter */ );

    /// Acquire synchronized camsci1 and camwfs means for the current DM state.
    /** \returns 0 on success, 1 if stopped, -1 on error
     */
    int acquire( bool                            settle,  /**< [in] discard settle frames first */
                 mx::improc::eigenImage<double> &sciMean, /**< [out] the camsci1 mean */
                 mx::improc::eigenImage<double> &wfsMean, /**< [out] the camwfs mean */
                 timingRecord                   &rec /**< [in.out] camera fields are set */ );

    /// Restore the DM channel to its initial content.
    void restoreDm();

    /// Write the response cubes for completed planes, and the timing table.
    /** \returns 0 on success, -1 on error
     */
    int writeOutput( bool final /**< [in] true for the final write, false for a checkpoint */ );

    /// Build the FITS header common to the output files.
    mx::fits::fitsHeader<> outputHeader( bool final /**< [in] true for the final write */ );

    /// Get the output file path for a suffix.
    std::string outputPath( const std::string &suffix /**< [in] appended to the prefix */ ) const;

    /// The signal handler, which requests a stop.
    static void handleSignal( int sig /**< [in] the signal number */ );

    /// Install the signal handler for SIGINT, SIGTERM, and SIGHUP.
    static void installSignalHandlers();
};

inline xlinqCal::~xlinqCal() noexcept
{
    try
    {
        stopWfsThread();
        restoreDm();
    }
    catch( ... )
    {
    }
}

inline std::vector<int> xlinqCal::signSequence( unsigned nRepeat )
{
    std::vector<int> signs;

    for( unsigned r = 0; r < nRepeat; ++r )
    {
        int first = ( r % 2 == 0 ) ? 1 : -1;
        signs.push_back( first );
        signs.push_back( -first );
    }

    return signs;
}

inline unsigned xlinqCal::settleCount( const std::vector<int> &signs )
{
    unsigned n = 0;

    for( size_t i = 0; i < signs.size(); ++i )
    {
        if( i == 0 || signs[i] != signs[i - 1] )
        {
            ++n;
        }
    }

    return n;
}

inline std::string xlinqCal::isoTime( const timespec &ts )
{
    tm uttime;
    gmtime_r( &ts.tv_sec, &uttime );

    char buf[64];
    strftime( buf, sizeof( buf ), "%Y-%m-%dT%H:%M:%S", &uttime );

    char out[80];
    snprintf( out, sizeof( out ), "%s.%03ld", buf, ts.tv_nsec / 1000000 );

    return out;
}

inline double xlinqCal::nowSec()
{
    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );

    return ts.tv_sec + ts.tv_nsec / 1e9;
}

inline void xlinqCal::requestStop()
{
    s_stop = 1;
}

inline void xlinqCal::clearStop()
{
    s_stop = 0;
}

inline int xlinqCal::nDone() const
{
    return m_nDone;
}

inline const std::vector<timingRecord> &xlinqCal::timing() const
{
    return m_timing;
}

inline const mx::improc::eigenCube<float> &xlinqCal::sciResp() const
{
    return m_sciResp;
}

inline const mx::improc::eigenCube<float> &xlinqCal::wfsResp() const
{
    return m_wfsResp;
}

inline void xlinqCal::setupConfig()
{
    using mx::app::argType;

    config.add( "dm.channel",
                "",
                "dm.channel",
                argType::Required,
                "dm",
                "channel",
                false,
                "string",
                "The DM channel stream to poke. Default dm01disp04." );
    config.add( "dm.comb",
                "",
                "dm.comb",
                argType::Required,
                "dm",
                "comb",
                false,
                "string",
                "The combined DM stream whose update confirms a poke. Default dm01disp." );
    config.add( "dm.wait",
                "",
                "dm.wait",
                argType::Required,
                "dm",
                "wait",
                false,
                "float",
                "Timeout [s] for the combined DM stream to update after a poke. Default 2." );
    config.add( "dm.projTol",
                "",
                "dm.projTol",
                argType::Required,
                "dm",
                "projTol",
                false,
                "float",
                "Warn if the combined DM change projected on the command differs from 1 by more than this. "
                "Default 0.25." );

    config.add( "poke.file",
                "",
                "poke.file",
                argType::Required,
                "poke",
                "file",
                false,
                "string",
                "FITS cube of DM shapes (rows x cols x Npoke), each normalized to 1 micron rms. Required." );
    config.add( "poke.amp",
                "",
                "poke.amp",
                argType::Required,
                "poke",
                "amp",
                false,
                "float",
                "Poke amplitude [microns rms]. Required." );
    config.add( "poke.first",
                "",
                "poke.first",
                argType::Required,
                "poke",
                "first",
                false,
                "int",
                "First poke index to measure. Default 0." );
    config.add( "poke.last",
                "",
                "poke.last",
                argType::Required,
                "poke",
                "last",
                false,
                "int",
                "Last poke index to measure, -1 for the last in the file. Default -1." );

    config.add( "sci.shmim",
                "",
                "sci.shmim",
                argType::Required,
                "sci",
                "shmim",
                false,
                "string",
                "The science camera stream. Default camsci1." );
    config.add( "sci.nSettle",
                "",
                "sci.nSettle",
                argType::Required,
                "sci",
                "nSettle",
                false,
                "int",
                "Frames discarded after the first frame following a poke. Default 1." );
    config.add( "sci.nFrames",
                "",
                "sci.nFrames",
                argType::Required,
                "sci",
                "nFrames",
                false,
                "int",
                "Frames averaged per poke sign. Default 10." );
    config.add( "sci.wait",
                "",
                "sci.wait",
                argType::Required,
                "sci",
                "wait",
                false,
                "float",
                "Timeout [s] for each science frame. Default 5." );

    config.add( "wfs.shmim",
                "",
                "wfs.shmim",
                argType::Required,
                "wfs",
                "shmim",
                false,
                "string",
                "The WFS camera stream. Default camwfs." );

    config.add( "seq.nRepeat",
                "",
                "seq.nRepeat",
                argType::Required,
                "seq",
                "nRepeat",
                false,
                "int",
                "Number of +/- pairs per poke, alternating +- and -+. Default 2." );

    config.add( "out.dir",
                "",
                "out.dir",
                argType::Required,
                "out",
                "dir",
                false,
                "string",
                "Output directory. Default is the current directory." );
    config.add( "out.prefix",
                "",
                "out.prefix",
                argType::Required,
                "out",
                "prefix",
                false,
                "string",
                "Output file prefix. Default xlinqCal_<UTC timestamp>." );
    config.add( "out.checkpointMin",
                "",
                "out.checkpointMin",
                argType::Required,
                "out",
                "checkpointMin",
                false,
                "float",
                "Minutes between checkpoint writes, 0 disables. Default 10." );

    config.add( "preflight.time",
                "",
                "preflight.time",
                argType::Required,
                "preflight",
                "time",
                false,
                "float",
                "Seconds to measure camera frame rates before starting. Default 2." );

    config.add( "yes", "y", "yes", argType::True, "", "", false, "bool", "Skip the confirmation prompt." );
}

inline void xlinqCal::loadConfig()
{
    config( m_dmChannel, "dm.channel" );
    config( m_dmComb, "dm.comb" );
    config( m_dmWait, "dm.wait" );
    config( m_dmProjTol, "dm.projTol" );

    config( m_pokeFile, "poke.file" );
    config( m_pokeAmp, "poke.amp" );
    config( m_pokeFirst, "poke.first" );
    config( m_pokeLast, "poke.last" );

    config( m_sciName, "sci.shmim" );
    config( m_sciSettle, "sci.nSettle" );
    config( m_sciFrames, "sci.nFrames" );
    config( m_sciWait, "sci.wait" );

    config( m_wfsName, "wfs.shmim" );

    config( m_nRepeat, "seq.nRepeat" );

    config( m_outDir, "out.dir" );
    config( m_outPrefix, "out.prefix" );
    config( m_checkpointMin, "out.checkpointMin" );

    config( m_preflightTime, "preflight.time" );

    config( m_yes, "yes" );
}

inline int xlinqCal::execute()
{
    clearStop();
    installSignalHandlers();

    if( m_pokeAmp == 0 )
    {
        logError( "poke.amp is required and must be nonzero" );
        return -1;
    }

    if( m_sciFrames == 0 || m_nRepeat == 0 )
    {
        logError( "sci.nFrames and seq.nRepeat must be at least 1" );
        return -1;
    }

    if( loadPokes() < 0 || openStreams() < 0 )
    {
        return -1;
    }

    int rv = preflight();
    if( rv != 0 )
    {
        return rv;
    }

    if( setupOutput() < 0 )
    {
        return -1;
    }

    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    m_dateBeg = isoTime( ts );

    try
    {
        startWfsThread();
        rv = run();
    }
    catch( const std::exception &e )
    {
        logError( std::string( "exception: " ) + e.what() );
        rv = -1;
    }

    restoreDm();
    stopWfsThread();

    clock_gettime( CLOCK_REALTIME, &ts );
    m_dateEnd = isoTime( ts );

    if( writeOutput( true ) < 0 )
    {
        rv = -1;
    }

    if( rv == 0 )
    {
        log( "complete: " + std::to_string( m_nDone ) + " pokes measured" );
    }
    else if( rv > 0 )
    {
        log( "stopped by request after " + std::to_string( m_nDone ) +
             " pokes; resume with poke.first=" + std::to_string( m_pokeFirst + m_nDone ) );
    }
    else
    {
        logError( "ended on error after " + std::to_string( m_nDone ) +
                  " pokes; resume with poke.first=" + std::to_string( m_pokeFirst + m_nDone ) );
    }

    return rv;
}

inline void xlinqCal::log( const std::string &msg )
{
    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );

    std::string line = isoTime( ts ) + " " + msg;
    std::cout << line << std::endl;

    if( m_logFile.is_open() )
    {
        m_logFile << line << std::endl;
    }
}

inline void xlinqCal::logError( const std::string &msg )
{
    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );

    std::string line = isoTime( ts ) + " ERROR: " + msg;
    std::cerr << line << std::endl;

    if( m_logFile.is_open() )
    {
        m_logFile << line << std::endl;
    }
}

inline int xlinqCal::loadPokes()
{
    if( m_pokeFile.empty() )
    {
        logError( "poke.file is required" );
        return -1;
    }

    mx::fits::fitsFile<float> ff;
    if( ff.read( m_pokes, m_pokeFile ) != mx::error_t::noerror )
    {
        logError( "could not read poke file " + m_pokeFile );
        return -1;
    }

    if( m_pokeLast < 0 )
    {
        m_pokeLast = m_pokes.planes() - 1;
    }

    if( m_pokeFirst < 0 || m_pokeFirst > m_pokeLast || m_pokeLast >= m_pokes.planes() )
    {
        logError( "invalid poke range " + std::to_string( m_pokeFirst ) + "-" + std::to_string( m_pokeLast ) + " for " +
                  std::to_string( m_pokes.planes() ) + " pokes" );
        return -1;
    }

    // Report shapes that are not normalized as expected, but don't refuse.
    for( int k = m_pokeFirst; k <= m_pokeLast; ++k )
    {
        auto   shape = m_pokes.image( k );
        double ms    = shape.square().sum() / shape.size();

        if( ms <= 0 )
        {
            logError( "poke " + std::to_string( k ) + " is all zeros" );
            return -1;
        }

        double rms = sqrt( ms );
        if( fabs( rms - 1 ) > 0.01 )
        {
            log( "warning: poke " + std::to_string( k ) + " has rms " + std::to_string( rms ) + ", expected 1" );
        }
    }

    return 0;
}

inline int xlinqCal::openStreams()
{
    try
    {
        m_dm.open( m_dmChannel );
        m_dm.passive( false );
    }
    catch( const std::exception &e )
    {
        logError( "could not open DM channel " + m_dmChannel + " as float: " + e.what() );
        return -1;
    }

    if( m_dm.rows() != m_pokes.rows() || m_dm.cols() != m_pokes.cols() )
    {
        logError( "DM channel is " + std::to_string( m_dm.rows() ) + "x" + std::to_string( m_dm.cols() ) +
                  " but pokes are " + std::to_string( m_pokes.rows() ) + "x" + std::to_string( m_pokes.cols() ) );
        return -1;
    }

    m_dmInitial = m_dm();
    m_dmLast    = m_dmInitial;

    if( m_dmInitial.abs().maxCoeff() > 0 )
    {
        log( "warning: " + m_dmChannel + " is not zero at startup; pokes are added to it and it will be restored" );
    }

    try
    {
        m_comb.open( m_dmComb );
        m_sci.open( m_sciName );
        m_wfs.open( m_wfsName );
    }
    catch( const std::exception &e )
    {
        logError( e.what() );
        return -1;
    }

    if( m_comb.rows() != m_dm.rows() || m_comb.cols() != m_dm.cols() )
    {
        logError( "combined DM stream " + m_dmComb + " size does not match " + m_dmChannel );
        return -1;
    }

    m_sciResp.resize( m_sci.rows(), m_sci.cols(), m_pokeLast - m_pokeFirst + 1 );
    m_wfsResp.resize( m_wfs.rows(), m_wfs.cols(), m_pokeLast - m_pokeFirst + 1 );
    m_sciResp.setZero();
    m_wfsResp.setZero();

    return 0;
}

inline int xlinqCal::preflight()
{
    uint64_t sci0 = m_sci.cnt0();
    uint64_t wfs0 = m_wfs.cnt0();
    double   t0   = nowSec();

    std::this_thread::sleep_for( std::chrono::duration<double>( m_preflightTime ) );

    double dt = nowSec() - t0;
    m_sciFps  = ( m_sci.cnt0() - sci0 ) / dt;
    m_wfsFps  = ( m_wfs.cnt0() - wfs0 ) / dt;

    if( m_sciFps <= 0 )
    {
        logError( m_sciName + " is not updating (check camera and shmimTCP)" );
        return -1;
    }

    if( m_wfsFps <= 0 )
    {
        logError( m_wfsName + " is not updating" );
        return -1;
    }

    std::vector<int> signs         = signSequence( m_nRepeat );
    int              nPoke         = m_pokeLast - m_pokeFirst + 1;
    double           framesPerPoke = settleCount( signs ) * ( 1.0 + m_sciSettle ) + signs.size() * m_sciFrames;
    double           hours         = nPoke * framesPerPoke / m_sciFps / 3600.;
    double           gbytes        = nPoke * 4.0 * ( m_sci.rows() * m_sci.cols() + m_wfs.rows() * m_wfs.cols() ) / 1e9;
    std::string      seq;

    for( int s : signs )
    {
        seq += ( s > 0 ) ? '+' : '-';
    }

    std::ostringstream os;
    os << "xlinqCal " << XLINQCAL_GIT_VERSION << "\n";
    os << "  DM:      " << m_dmChannel << " (" << m_dm.rows() << "x" << m_dm.cols() << "), confirm on " << m_dmComb
       << "\n";
    os << "  pokes:   " << m_pokeFile << " [" << m_pokeFirst << "-" << m_pokeLast << "] = " << nPoke << " pokes, amp "
       << m_pokeAmp << " um rms\n";
    os << "  sci:     " << m_sciName << " " << m_sci.rows() << "x" << m_sci.cols() << " type " << (int)m_sci.datatype()
       << " at " << m_sciFps << " fps\n";
    os << "  wfs:     " << m_wfsName << " " << m_wfs.rows() << "x" << m_wfs.cols() << " type " << (int)m_wfs.datatype()
       << " at " << m_wfsFps << " fps (~" << m_wfsFps / m_sciFps << " per sci frame)\n";
    os << "  timing:  discard 1+" << m_sciSettle << ", average " << m_sciFrames << " sci frames, sequence " << seq
       << "\n";
    os << "  estimate: " << hours << " h, " << gbytes << " GB output";
    log( os.str() );

    if( m_yes )
    {
        return 0;
    }

    std::cout << "Proceed? [y/N] " << std::flush;

    std::string answer;
    std::getline( std::cin, answer );

    if( answer != "y" && answer != "Y" && answer != "yes" )
    {
        log( "declined by operator" );
        return 1;
    }

    return 0;
}

inline int xlinqCal::setupOutput()
{
    if( m_outPrefix.empty() )
    {
        timespec ts;
        clock_gettime( CLOCK_REALTIME, &ts );

        tm uttime;
        gmtime_r( &ts.tv_sec, &uttime );

        char buf[64];
        strftime( buf, sizeof( buf ), "xlinqCal_%Y%m%dT%H%M%S", &uttime );
        m_outPrefix = buf;
    }

    std::error_code ec;
    std::filesystem::create_directories( m_outDir, ec );

    if( ec )
    {
        logError( "could not create output directory " + m_outDir + ": " + ec.message() );
        return -1;
    }

    m_logFile.open( outputPath( ".log" ), std::ios::app );

    if( !m_logFile.is_open() )
    {
        logError( "could not open log file " + outputPath( ".log" ) );
        return -1;
    }

    log( "output prefix " + outputPath( "" ) );

    return 0;
}

inline void xlinqCal::wfsThreadExec()
{
    mx::improc::eigenImage<double> frame;

    while( m_wfsRun )
    {
        waitResult wr = m_wfs.wait( 0.5 );

        if( wr == waitResult::timeout )
        {
            continue;
        }

        if( wr == waitResult::error )
        {
            m_wfsErrno = errno ? errno : -1;
            return;
        }

        uint64_t cnt0;
        bool     ok = m_wfs.copy( frame, cnt0 );

        if( cnt0 == m_wfsLastSeen )
        {
            continue; // A repeated post for a frame already seen
        }

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_wfsMutex );

            if( m_wfsGate )
            {
                if( !m_wfsGateFirst && cnt0 > m_wfsLastSeen + 1 )
                {
                    m_wfsMissed += cnt0 - m_wfsLastSeen - 1;
                }

                m_wfsGateFirst = false;

                if( ok )
                {
                    if( m_wfsN == 0 )
                    {
                        m_wfsCnt0First = cnt0;
                    }

                    m_wfsSum += frame;
                    m_wfsCnt0Last = cnt0;
                    ++m_wfsN;
                }
                else
                {
                    ++m_wfsTorn;
                }
            }
        }

        m_wfsLastSeen = cnt0;
    }
}

inline void xlinqCal::startWfsThread()
{
    m_wfsSum.resize( m_wfs.rows(), m_wfs.cols() );
    m_wfsSum.setZero();
    m_wfsGate     = false;
    m_wfsErrno    = 0;
    m_wfsLastSeen = m_wfs.cnt0();
    m_wfsRun      = true;
    m_wfsThread   = std::thread( &xlinqCal::wfsThreadExec, this );
}

inline void xlinqCal::stopWfsThread()
{
    m_wfsRun = false;

    if( m_wfsThread.joinable() )
    {
        m_wfsThread.join();
    }
}

inline int xlinqCal::run()
{
    int    nPoke    = m_pokeLast - m_pokeFirst + 1;
    double tStart   = nowSec();
    double tLastChk = tStart;

    m_sciLastCnt0 = m_sci.cnt0();

    for( int k = m_pokeFirst; k <= m_pokeLast; ++k )
    {
        if( s_stop )
        {
            return 1;
        }

        int rv = measurePoke( k );

        if( rv != 0 )
        {
            return rv;
        }

        ++m_nDone;

        const timingRecord &last    = m_timing.back();
        double              elapsed = nowSec() - tStart;
        double              eta     = elapsed / m_nDone * ( nPoke - m_nDone );

        std::ostringstream os;
        os << "poke " << k << " (" << m_nDone << "/" << nPoke << ") wfs frames/window " << last[timingCol::wfsN]
           << ", elapsed " << elapsed / 60 << " min, remaining " << eta / 60 << " min";
        log( os.str() );

        if( m_checkpointMin > 0 && nowSec() - tLastChk > m_checkpointMin * 60 && m_nDone < nPoke )
        {
            writeOutput( false );
            tLastChk = nowSec();
        }
    }

    return 0;
}

inline int xlinqCal::measurePoke( int k )
{
    std::vector<int> signs = signSequence( m_nRepeat );

    mx::improc::eigenImage<double> sciMean, wfsMean;
    mx::improc::eigenImage<double> sciAcc( m_sci.rows(), m_sci.cols() );
    mx::improc::eigenImage<double> wfsAcc( m_wfs.rows(), m_wfs.cols() );
    sciAcc.setZero();
    wfsAcc.setZero();

    for( size_t i = 0; i < signs.size(); ++i )
    {
        timingRecord rec;
        rec.fill( std::numeric_limits<double>::quiet_NaN() );
        rec[timingCol::poke]   = k;
        rec[timingCol::repeat] = i / 2;
        rec[timingCol::sign]   = signs[i];

        mx::improc::eigenImage<float> cmd = m_dmInitial + ( signs[i] * m_pokeAmp ) * m_pokes.image( k );

        bool settle = !( cmd == m_dmLast ).all();

        if( settle )
        {
            int rv = applyCommand( cmd, rec );
            if( rv != 0 )
            {
                return rv;
            }
        }

        int rv = acquire( settle, sciMean, wfsMean, rec );
        if( rv != 0 )
        {
            return rv;
        }

        sciAcc += static_cast<double>( signs[i] ) * sciMean;
        wfsAcc += static_cast<double>( signs[i] ) * wfsMean;

        m_timing.push_back( rec );
    }

    // (mean(+) - mean(-))/2, averaged over repeats
    double nAcq                        = signs.size();
    m_sciResp.image( k - m_pokeFirst ) = ( sciAcc / nAcq ).cast<float>();
    m_wfsResp.image( k - m_pokeFirst ) = ( wfsAcc / nAcq ).cast<float>();

    return 0;
}

inline int xlinqCal::applyCommand( const mx::improc::eigenImage<float> &cmd, timingRecord &rec )
{
    mx::improc::eigenImage<double> before, after;
    uint64_t                       cnt0Before, cnt0After;

    m_comb.copy( before, cnt0Before );
    m_comb.flush();

    m_dm         = cmd;
    m_dmWritten  = true;
    double tWait = nowSec() + m_dmWait;

    do
    {
        if( s_stop )
        {
            return 1;
        }

        if( nowSec() > tWait )
        {
            logError( "timeout waiting for " + m_dmComb + " to update" );
            return -1;
        }

        if( m_comb.wait( 0.1 ) == waitResult::error )
        {
            logError( "error waiting on " + m_dmComb );
            return -1;
        }
    } while( m_comb.cnt0() == cnt0Before );

    m_comb.copy( after, cnt0After );

    mx::improc::eigenImage<double> dcmd = ( cmd - m_dmLast ).cast<double>();
    double                         proj = ( ( after - before ) * dcmd ).sum() / dcmd.square().sum();

    rec[timingCol::dmCnt0] = cnt0After;
    rec[timingCol::dmProj] = proj;

    if( fabs( proj - 1 ) > m_dmProjTol )
    {
        log( "warning: " + m_dmComb + " change projected on the command is " + std::to_string( proj ) +
             " (expected 1) at poke " + std::to_string( (int)rec[timingCol::poke] ) );
    }

    m_dmLast = cmd;

    return 0;
}

inline int xlinqCal::waitSciFrame( mx::improc::eigenImage<double> *frame, uint64_t &cnt0 )
{
    double tWait = nowSec() + m_sciWait;

    while( true )
    {
        if( s_stop )
        {
            return 1;
        }

        if( m_wfsErrno != 0 )
        {
            logError( "WFS thread error waiting on " + m_wfsName );
            return -1;
        }

        if( nowSec() > tWait )
        {
            logError( "timeout waiting for " + m_sciName );
            return -1;
        }

        waitResult wr = m_sci.wait( 0.1 );

        if( wr == waitResult::error )
        {
            logError( "error waiting on " + m_sciName );
            return -1;
        }

        if( wr == waitResult::timeout || m_sci.cnt0() == m_sciLastCnt0 )
        {
            continue;
        }

        if( frame == nullptr )
        {
            cnt0 = m_sci.cnt0();
        }
        else if( !m_sci.copy( *frame, cnt0 ) )
        {
            continue; // Torn: wait for the next frame, the gap is counted as missed
        }

        m_sciLastCnt0 = cnt0;

        return 0;
    }
}

inline int xlinqCal::acquire( bool                            settle,
                              mx::improc::eigenImage<double> &sciMean,
                              mx::improc::eigenImage<double> &wfsMean,
                              timingRecord                   &rec )
{
    uint64_t cnt0;
    int      rv;

    if( settle )
    {
        // Discard the first frame following the poke, then nSettle more.
        m_sci.flush();
        m_sciLastCnt0 = m_sci.cnt0();

        for( unsigned n = 0; n < 1 + m_sciSettle; ++n )
        {
            if( ( rv = waitSciFrame( nullptr, cnt0 ) ) != 0 )
            {
                return rv;
            }
        }
    }

    // Open the gate: the next camsci1 exposure starts now.
    { //mutex scope
        std::lock_guard<std::mutex> lock( m_wfsMutex );
        m_wfsSum.setZero();
        m_wfsN         = 0;
        m_wfsMissed    = 0;
        m_wfsTorn      = 0;
        m_wfsGateFirst = true;
        m_wfsGate      = true;
    }

    rec[timingCol::tStart] = nowSec();

    mx::improc::eigenImage<double> frame;
    mx::improc::eigenImage<double> sciSum( m_sci.rows(), m_sci.cols() );
    sciSum.setZero();

    uint64_t sciMissed = 0;
    uint64_t prev      = m_sciLastCnt0;

    for( unsigned n = 0; n < m_sciFrames; ++n )
    {
        if( ( rv = waitSciFrame( &frame, cnt0 ) ) != 0 )
        {
            std::lock_guard<std::mutex> lock( m_wfsMutex );
            m_wfsGate = false;
            return rv;
        }

        if( n == 0 )
        {
            rec[timingCol::sciCnt0First] = cnt0;
        }
        else if( cnt0 > prev + 1 )
        {
            sciMissed += cnt0 - prev - 1;
        }

        prev = cnt0;
        sciSum += frame;
    }

    { //mutex scope
        std::lock_guard<std::mutex> lock( m_wfsMutex );
        m_wfsGate = false;

        rec[timingCol::wfsN]         = m_wfsN;
        rec[timingCol::wfsMissed]    = m_wfsMissed;
        rec[timingCol::wfsTorn]      = m_wfsTorn;
        rec[timingCol::wfsCnt0First] = m_wfsCnt0First;
        rec[timingCol::wfsCnt0Last]  = m_wfsCnt0Last;

        if( m_wfsN > 0 )
        {
            wfsMean = m_wfsSum / static_cast<double>( m_wfsN );
        }
    }

    rec[timingCol::tEnd]        = nowSec();
    rec[timingCol::sciCnt0Last] = prev;
    rec[timingCol::sciN]        = m_sciFrames;
    rec[timingCol::sciMissed]   = sciMissed;

    if( rec[timingCol::wfsN] == 0 )
    {
        logError( "no " + m_wfsName + " frames received during the averaging window" );
        return -1;
    }

    sciMean = sciSum / static_cast<double>( m_sciFrames );

    return 0;
}

inline void xlinqCal::restoreDm()
{
    if( !m_dmWritten )
    {
        return;
    }

    m_dm        = m_dmInitial;
    m_dmLast    = m_dmInitial;
    m_dmWritten = false;

    log( "restored " + m_dmChannel );
}

inline mx::fits::fitsHeader<> xlinqCal::outputHeader( bool final )
{
    std::string seq;
    for( int s : signSequence( m_nRepeat ) )
    {
        seq += ( s > 0 ) ? '+' : '-';
    }

    mx::fits::fitsHeader<> head;
    head.append( "XLQVERS", std::string( XLINQCAL_GIT_VERSION ), "xlinqCal version" );
    head.append( "COMPLETE", final && m_nDone == m_pokeLast - m_pokeFirst + 1, "all requested pokes measured" );
    head.append( "RESPDEF", std::string( "(mean(+)-mean(-))/2" ), "not divided by POKEAMP" );
    head.append( "POKEFILE", m_pokeFile, "poke shapes, 1 um rms each" );
    head.append( "POKEAMP", m_pokeAmp, "poke amplitude [um rms]" );
    head.append( "POKE0", m_pokeFirst, "poke index of first plane" );
    head.append( "NPOKEREQ", m_pokeLast - m_pokeFirst + 1, "number of pokes requested" );
    head.append( "NPOKE", m_nDone, "number of planes measured" );
    head.append( "NSETTLE", m_sciSettle, "sci frames discarded after the first post-poke frame" );
    head.append( "NFRAMES", m_sciFrames, "sci frames averaged per sign" );
    head.append( "NREPEAT", m_nRepeat, "+/- pairs per poke" );
    head.append( "SEQUENCE", seq, "sign sequence per poke" );
    head.append( "DMCHAN", m_dmChannel, "DM channel poked" );
    head.append( "DMCOMB", m_dmComb, "combined DM stream" );
    head.append( "SCISTRM", m_sciName, "science camera stream" );
    head.append( "SCIFPS", m_sciFps, "measured sci frame rate [Hz]" );
    head.append( "SCITYPE", (int)m_sci.datatype(), "sci ImageStreamIO datatype" );
    head.append( "WFSSTRM", m_wfsName, "WFS camera stream" );
    head.append( "WFSFPS", m_wfsFps, "measured WFS frame rate [Hz]" );
    head.append( "WFSTYPE", (int)m_wfs.datatype(), "WFS ImageStreamIO datatype" );
    head.append( "DATE-BEG", m_dateBeg, "UTC start of measurement" );
    head.append( "DATE-END", m_dateEnd.empty() ? std::string( "running" ) : m_dateEnd, "UTC end of measurement" );

    return head;
}

inline std::string xlinqCal::outputPath( const std::string &suffix ) const
{
    return ( std::filesystem::path( m_outDir ) / ( m_outPrefix + suffix ) ).string();
}

inline int xlinqCal::writeOutput( bool final )
{
    if( m_nDone == 0 )
    {
        log( "no completed pokes, nothing written" );
        return 0;
    }

    int rv = 0;

    mx::fits::fitsHeader<> head = outputHeader( final );

    // Planes are contiguous, so the first m_nDone planes are written directly.
    mx::fits::fitsFile<float> ff;
    std::string               sciFile = outputPath( "_" + m_sciName + "_resp.fits" );

    if( ff.write( sciFile, m_sciResp.data(), m_sciResp.rows(), m_sciResp.cols(), m_nDone, head ) !=
        mx::error_t::noerror )
    {
        logError( "could not write " + sciFile );
        rv = -1;
    }

    std::string wfsFile = outputPath( "_" + m_wfsName + "_resp.fits" );

    if( ff.write( wfsFile, m_wfsResp.data(), m_wfsResp.rows(), m_wfsResp.cols(), m_nDone, head ) !=
        mx::error_t::noerror )
    {
        logError( "could not write " + wfsFile );
        rv = -1;
    }

    static const char *colNames[timingCol::nCols] = { "POKE",
                                                      "REPEAT",
                                                      "SIGN",
                                                      "DMCNT0",
                                                      "DMPROJ",
                                                      "SCICNT0A",
                                                      "SCICNT0B",
                                                      "SCIN",
                                                      "SCIMISS",
                                                      "WFSCNT0A",
                                                      "WFSCNT0B",
                                                      "WFSN",
                                                      "WFSMISS",
                                                      "WFSTORN",
                                                      "TSTART",
                                                      "TEND" };

    mx::improc::eigenImage<double> table( timingCol::nCols, m_timing.size() );

    for( size_t r = 0; r < m_timing.size(); ++r )
    {
        for( int c = 0; c < timingCol::nCols; ++c )
        {
            table( c, r ) = m_timing[r][c];
        }
    }

    for( int c = 0; c < timingCol::nCols; ++c )
    {
        head.append( "TCOL" + std::to_string( c ), std::string( colNames[c] ), "timing table column" );
    }

    mx::fits::fitsFile<double> fd;
    std::string                timingFile = outputPath( "_timing.fits" );

    if( fd.write( timingFile, table, head ) != mx::error_t::noerror )
    {
        logError( "could not write " + timingFile );
        rv = -1;
    }

    if( rv == 0 )
    {
        log( std::string( final ? "wrote " : "checkpoint: wrote " ) + std::to_string( m_nDone ) + " planes to " +
             outputPath( "_*" ) );
    }

    return rv;
}

inline void xlinqCal::handleSignal( int sig )
{
    static_cast<void>( sig );
    s_stop = 1;
}

inline void xlinqCal::installSignalHandlers()
{
    struct sigaction act{};
    act.sa_handler = &xlinqCal::handleSignal;
    sigemptyset( &act.sa_mask );

    sigaction( SIGINT, &act, nullptr );
    sigaction( SIGTERM, &act, nullptr );
    sigaction( SIGHUP, &act, nullptr );
}

} // namespace xlinqCal

#endif // xlinqCal_hpp
