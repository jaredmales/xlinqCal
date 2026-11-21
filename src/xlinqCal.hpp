/** \file xlinqCal.hpp
 * \brief The xlinqCal application: measure synchronized camwfs and camsci1 responses to DM pokes.
 *
 * \ingroup xlinqCal_files
 */

#ifndef xlinqCal_hpp
#define xlinqCal_hpp

#include <limits>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <vector>

#include <mx/app/application.hpp>
#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>
#include <mx/improc/milkImage.hpp>
#include <mx/ioutils/fits/fitsFile.hpp>

#include "indiClient.hpp"
#include "isioStream.hpp"
#include "wfsChannel.hpp"

#ifndef XLINQCAL_GIT_VERSION
    #define XLINQCAL_GIT_VERSION "unknown"
#endif

/** \defgroup xlinqCal xlinqCal
 * \brief Standalone MagAO-X calibration of synchronized WFS and camsci1 poke responses.
 *
 * The DM is poked by writing `amp * shape` to a DM channel (default `dm01disp04`), and the
 * camwfs and camsci1 frames taken while the poke is on the DM are averaged. Frames are
 * synchronized to camsci1: after a poke, the first camsci1 frame and `sci.nSettle` more are
 * discarded, then camwfs frames are averaged between the posts that bound `sci.nFrames` averaged
 * camsci1 frames. The response for each camera is `(mean(+) - mean(-))/2`, averaged over
 * `seq.nRepeat` repeats with alternating sign order (`+-`, `-+`, ...). The symmetric mean
 * `(mean(+) + mean(-))/2` is also saved. Zero-poke reference acquisitions are interleaved every
 * `ref.every` pokes, and the even part `mean - ref` is formed with the reference linearly
 * interpolated to each poke's center time, which cancels linear drift. Several WFS streams can be
 * gated together (`wfs.shmim`), e.g. calibrated frames `aol1_imWFS2pub` and the loop's modal
 * coefficients `aol1_modevalWFSpub`. Streams listed in `snap.shmim` (e.g. `aol1_wfsref`,
 * `aol1_wfsmask`) are saved at the start, and again at the end if they changed.
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
    poke,          ///< Poke index in the input cube, NaN for a reference.
    repeat,        ///< Repeat index, or the reference index for a reference.
    sign,          ///< Poke sign, +1 or -1, or 0 for a reference.
    dmCnt0,        ///< cnt0 of the combined DM stream when the poke was confirmed, NaN if no DM write.
    dmProj,        ///< Projection of the combined DM change onto the commanded change, NaN if no DM write.
    sciCnt0First,  ///< cnt0 of the first averaged camsci1 frame.
    sciCnt0Last,   ///< cnt0 of the last averaged camsci1 frame.
    sciN,          ///< Number of camsci1 frames averaged.
    sciMissed,     ///< camsci1 frames skipped (cnt0 gaps) during averaging.
    tStart,        ///< Gate open time, Unix seconds.
    tEnd,          ///< Gate close time, Unix seconds.
    sciAtimeFirst, ///< Acquisition time (stream atime) of the first averaged camsci1 frame, Unix seconds.
    sciAtimeLast,  ///< Acquisition time (stream atime) of the last averaged camsci1 frame, Unix seconds.
    sciMax,        ///< Maximum pixel value over the averaged camsci1 frames.
    nCommon        ///< The number of common columns; per-WFS-stream columns follow.
};
} // namespace timingCol

/// Per-WFS-stream column offsets in the timing table, after the common columns.
namespace wfsCol
{
enum : int
{
    cnt0First,  ///< cnt0 of the first averaged frame.
    cnt0Last,   ///< cnt0 of the last averaged frame.
    n,          ///< Number of frames averaged.
    missed,     ///< Frames skipped (cnt0 gaps) while the gate was open.
    torn,       ///< Frames dropped because they changed during the copy.
    atimeFirst, ///< Acquisition time (stream atime) of the first averaged frame, Unix seconds.
    atimeLast,  ///< Acquisition time (stream atime) of the last averaged frame, Unix seconds.
    nCols       ///< The number of columns per WFS stream.
};
} // namespace wfsCol

/// Get the timing table column index of a field for WFS stream j.
inline size_t wfsColIndex( size_t j, /**< [in] the WFS stream index */
                           int    field /**< [in] the wfsCol field */ )
{
    return timingCol::nCommon + j * wfsCol::nCols + field;
}

/// One row of the timing table: timingCol::nCommon columns, then wfsCol::nCols per WFS stream.
typedef std::vector<double> timingRecord;

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

    std::string m_pokeFile; ///< FITS cube of DM shapes, rows x cols x Npoke, 1 micron rms (operator-supplied).

    float m_pokeAmp{ 0 }; ///< The poke amplitude, microns rms. Required, must be nonzero.

    int m_pokeFirst{ 0 }; ///< First poke index in m_pokeFile to measure.

    int m_pokeLast{ -1 }; ///< Last poke index in m_pokeFile to measure, -1 for the last in the file.

    std::string m_sciName{ "camsci1" }; ///< The science camera stream, which sets synchronization.

    unsigned m_sciSettle{ 1 }; ///< camsci1 frames discarded after the first frame following a poke.

    unsigned m_sciFrames{ 10 }; ///< camsci1 frames averaged per poke sign.

    double m_sciWait{ 5.0 }; ///< Timeout in seconds for each camsci1 frame.

    double m_sciLinearMax{ 37000 }; ///< camsci1 linearity limit [counts]; frames above it are flagged, 0 disables.

    std::vector<std::string> m_wfsNames{ "camwfs" }; ///< The WFS streams, averaged over the camsci1 windows.

    std::vector<std::string> m_snapNames; ///< Streams saved at the start (and end, if changed), e.g. aol1_wfsref.

    bool m_indiEnable{ false }; ///< If true, record INDI state and allow a dark; no INDI calls are made otherwise.

    std::string m_indiGet{ "getINDI" }; ///< The getINDI command.

    std::string m_indiSet{ "setINDI" }; ///< The setINDI command.

    double m_indiTimeout{ 8 }; ///< Wall-clock limit for each INDI call [s].

    double m_indiPoll{ 2 }; ///< Interval between INDI readbacks while waiting [s]; MagAOX asks for >= 2 s.

    std::vector<std::string> m_indiProps; ///< Extra `device.property` names recorded at start and end.

    std::string m_sciDevice{ "camsci1" }; ///< The INDI device of the science camera.

    unsigned m_darkN{ 0 }; ///< camsci1 frames averaged for the dark after the run, 0 for no dark.

    unsigned m_darkDiscard{ 2 }; ///< camsci1 frames discarded after the shutter closes.

    double m_darkWait{ 15 }; ///< Timeout [s] for the shutter to reach the requested state.

    unsigned m_nRepeat{ 2 }; ///< Number of +/- pairs per poke, with alternating order.

    unsigned m_refEvery{ 1 }; ///< Pokes between zero-poke reference acquisitions, 0 disables references.

    std::string m_outDir{ "." }; ///< Output directory, created if needed.

    std::string m_outPrefix{ "xlinqCal" }; ///< Output file prefix, before the timestamp.

    bool m_outTimestamp{ true }; ///< Append _<UTC timestamp> to the prefix.

    double m_checkpointMin{ 10 }; ///< Minutes between checkpoint writes of completed planes, 0 disables.

    double m_preflightTime{ 2.0 }; ///< Seconds to measure camera frame rates before starting.

    bool m_yes{ false }; ///< Skip the operator confirmation prompt.

    ///@}

    /** \name Streams and Data - Data
     * @{
     */

    isioStream m_sci; ///< The camsci1 stream, with claimed semaphore.

    std::vector<std::unique_ptr<wfsChannel>> m_wfs; ///< The WFS streams, each with its own reader thread.

    std::vector<std::unique_ptr<isioStream>> m_snap; ///< The snapshot streams, opened without semaphores.

    indiClient m_indi; ///< The INDI client (getINDI/setINDI).

    std::map<std::string, std::string> m_indiStart; ///< INDI state at the start, `device.property.element` to value.

    double m_sciExptime{ std::numeric_limits<double>::quiet_NaN() }; ///< camsci1 exposure time from INDI [s].

    double m_sciEmgain{ std::numeric_limits<double>::quiet_NaN() }; ///< camsci1 EM gain from INDI.

    bool m_shutterShut{ false }; ///< True if xlinqCal shut the camsci1 shutter and has not confirmed it open again.

    unsigned m_darkTaken{ 0 }; ///< Frames in the saved dark, 0 if none.

    std::vector<mx::improc::eigenImage<float>> m_snapStart; ///< Snapshot content at the start of measurement.

    isioStream m_comb; ///< The combined DM stream, with claimed semaphore.

    mx::improc::milkImage<float> m_dm; ///< The DM channel, written non-passively so dmcomb updates.

    mx::improc::eigenImage<float> m_dmInitial; ///< DM channel content at startup, restored on exit.

    mx::improc::eigenImage<float> m_dmLast; ///< The last command written to the DM channel.

    bool m_dmWritten{ false }; ///< True once the DM channel has been written, so it needs restoring.

    mx::improc::eigenCube<float> m_pokes; ///< The poke shapes, normalized to 1 micron rms.

    mx::fits::fitsHeader<> m_pokeHead; ///< The poke file's header, without its standard top, copied into _pokes.fits.

    camProducts m_sciProd; ///< The camsci1 per-poke products (the WFS products are in each wfsChannel).

    std::vector<double> m_pokeTime; ///< Center time of each completed poke, Unix seconds.

    std::vector<double> m_refTime; ///< Center time of each reference, Unix seconds.

    int m_nRef{ 0 }; ///< Number of completed references.

    bool m_firstAcq{ true }; ///< True until the first acquisition, which always settles to align the gate.

    int m_nDone{ 0 }; ///< Number of completed response planes.

    std::vector<timingRecord> m_timing; ///< One timing row per acquisition.

    double m_sciFps{ 0 }; ///< camsci1 frame rate measured during preflight.

    double m_sciPreMax{ 0 }; ///< Maximum camsci1 pixel value in a frame read during preflight.

    int m_nNonlin{ 0 }; ///< Number of acquisitions with a camsci1 frame above m_sciLinearMax.

    uint64_t m_sciLastCnt0{ 0 }; ///< cnt0 of the last camsci1 frame received.

    std::string m_dateBeg; ///< UTC start of measurement, ISO 8601.

    std::string m_dateEnd; ///< UTC end of measurement, ISO 8601.

    std::ofstream m_logFile; ///< Run log, opened in the output directory.

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

    /// Check whether a name is a string repeated twice, e.g. "abcabc".
    /** mxlib's ini parser joins the values of a key repeated within a section with no separator,
     * so a doubled stream name usually means a repeated key in the config file.
     */
    static bool looksDoubled( const std::string &name /**< [in] the name to check */ );

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

    /// Get the number of acquisitions with a camsci1 pixel above sci.linearMax.
    int nNonlin() const;

    /// Get the output path prefix, `out.dir/<prefix>`, set when the run starts.
    std::string outputPrefix() const;

    /// Get the timing table.
    const std::vector<timingRecord> &timing() const;

    /// Get the camsci1 per-poke products.
    const camProducts &sciProducts() const;

    /// Get the number of WFS streams.
    size_t nWfs() const;

    /// Get the per-poke products of WFS stream j.
    const camProducts &wfsProducts( size_t j /**< [in] the WFS stream index */ ) const;

    /// Get the number of completed references.
    int nRef() const;

    /// Find the references bracketing a time, and the linear interpolation weight.
    /** The interpolated reference is `(1-w1)*ref[i0] + w1*ref[i1]`. If t is not bracketed, the
     * nearest reference is used (i0 == i1, w1 = 0).
     *
     * \returns true if t is bracketed by two references
     * \returns false if the nearest reference is used, or if tRef is empty (i0 = i1 = 0)
     */
    static bool refWeights( const std::vector<double> &tRef, /**< [in] reference times, increasing */
                            double                     t,    /**< [in] the time to interpolate to */
                            size_t                    &i0,   /**< [out] index of the earlier reference */
                            size_t                    &i1,   /**< [out] index of the later reference */
                            double                    &w1 /**< [out] the weight of ref[i1] */ );

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

    /// Append the timestamp to the prefix, and refuse a prefix that is already used in the output directory.
    /** \returns 0 on success, -1 on error
     */
    int resolvePrefix();

    /// Check whether any file in the output directory is named `<prefix>_*` or `<prefix>.*`.
    /** \returns true if the prefix is in use, with one such file name in example
     */
    bool prefixInUse( std::string &example /**< [out] a file using the prefix */ ) const;

    /// Create the output directory and open the run log.
    /** \returns 0 on success, -1 on error
     */
    int setupOutput();

    /// Start the WFS reader threads.
    void startWfsThread();

    /// Stop and join the WFS reader threads.
    void stopWfsThread();

    /// Make a timing row of NaNs, sized for the WFS streams.
    timingRecord newRecord() const;

    /// Measure all pokes, with checkpoints.
    /** \returns 0 on success, 1 if stopped, -1 on error
     */
    int run();

    /// Measure the response for one poke, over all signs and repeats.
    /** \returns 0 on success, 1 if stopped, -1 on error
     */
    int measurePoke( int k /**< [in] the poke index in m_pokes */ );

    /// Measure a zero-poke reference and store it.
    /** \returns 0 on success, 1 if stopped, -1 on error
     */
    int measureRef();

    /// Set the DM command if it changed, then acquire synchronized means.
    /** Settles if the command changed, or on the first acquisition.
     *
     * \returns 0 on success, 1 if stopped, -1 on error
     */
    int setAndAcquire( const mx::improc::eigenImage<float>         &cmd,      /**< [in] the full channel command */
                       mx::improc::eigenImage<double>              &sciMean,  /**< [out] the camsci1 mean */
                       std::vector<mx::improc::eigenImage<double>> &wfsMeans, /**< [out] the WFS means */
                       timingRecord                                &rec /**< [in.out] timing fields are set */ );

    /// Form the even part `mean - ref` for completed pokes, with references interpolated in time.
    /** \returns the number of planes that used a one-sided (nearest) reference
     */
    int evenCube( mx::improc::eigenCube<float>       &even, /**< [out] the even part, m_nDone planes */
                  const mx::improc::eigenCube<float> &mean, /**< [in] the symmetric means */
                  const mx::improc::eigenCube<float> &ref /**< [in] the references */ ) const;

    /// Write the first nPlanes planes of a cube with the common header and a product description.
    /** \returns 0 on success, -1 on error
     */
    int writeCube( const std::string                  &suffix,  /**< [in] appended to the output prefix */
                   const mx::improc::eigenCube<float> &cube,    /**< [in] the cube */
                   int                                 nPlanes, /**< [in] the number of planes to write */
                   mx::fits::fitsHeader<>              head,    /**< [in] the common header (copied) */
                   const std::string                  &product /**< [in] the PRODUCT keyword value */ );

    /// Write a DM command and wait for the combined DM stream to update.
    /** \returns 0 on success, 1 if stopped, -1 on error
     */
    int applyCommand( const mx::improc::eigenImage<float> &cmd, /**< [in] the full channel command */
                      timingRecord                        &rec /**< [in.out] DM fields are set */ );

    /// Wait for the next new camsci1 frame, optionally copying it.
    /** \returns 0 on success, 1 if stopped, -1 on timeout or error
     */
    int waitSciFrame( mx::improc::eigenImage<double> *frame, /**< [out] the frame, or nullptr to only wait */
                      uint64_t                       &cnt0,  /**< [out] the frame counter */
                      timespec                       *atime = nullptr /**< [out] the frame's atime, if not null */ );

    /// Acquire synchronized camsci1 and WFS means for the current DM state.
    /** \returns 0 on success, 1 if stopped, -1 on error
     */
    int acquire( bool                                         settle,   /**< [in] discard settle frames first */
                 mx::improc::eigenImage<double>              &sciMean,  /**< [out] the camsci1 mean */
                 std::vector<mx::improc::eigenImage<double>> &wfsMeans, /**< [out] the WFS means */
                 timingRecord                                &rec /**< [in.out] camera fields are set */ );

    /// Copy the current content of each snapshot stream.
    /** \returns 0 on success, -1 on error
     */
    int readSnapshots( std::vector<mx::improc::eigenImage<float>> &ims /**< [out] one image per stream */ );

    /// Write one snapshot to `<prefix>_snap_<name><suffix>.fits`.
    /** \returns 0 on success, -1 on error
     */
    int writeSnapshot( size_t                               j,  /**< [in] the snapshot stream index */
                       const mx::improc::eigenImage<float> &im, /**< [in] the content */
                       const std::string                   &suffix /**< [in] e.g. "" or "_end" */ );

    /// Save the applied pokes, `poke.amp` times the requested planes, with the poke file's header.
    /** Written to `<prefix>_pokes.fits` before the first poke.
     *
     * \returns 0 on success, -1 on error
     */
    int savePokes();

    /// Save the start snapshots.
    /** \returns 0 on success, -1 on error
     */
    int saveStartSnapshots();

    /// Compare the end state of the snapshot streams with the start, and save any that changed.
    /** \returns 0 on success (changed or not), -1 on error
     */
    int checkEndSnapshots();

    /// The INDI properties recorded at start and end: the camera's standard ones, then indi.props.
    std::vector<std::string> indiPropList() const;

    /// Check INDI before the run: the camera device answers, and for a dark the shutter is ready and open.
    /** \returns 0 on success (or INDI disabled), -1 on error
     */
    int indiPreflight();

    /// Record the INDI properties to `<prefix><suffix>`, one `device.property.element=value` line each.
    /** Properties that can't be read are noted in the file and logged; they are not errors.
     * \returns 0 on success, -1 if the file can't be written
     */
    int indiRecord( const std::string                  &suffix, /**< [in] appended to the output prefix */
                    std::map<std::string, std::string> &vals /**< [out] `device.property.element` to value */ );

    /// Command the camsci1 shutter and wait for the readback.
    /** `shutter.toggle` On is shut and Off is open (MagAOX stdCamera); the readback is polled every
     * indi.poll seconds until it matches or dark.wait passes.
     * \returns 0 when the readback matches, -1 on error or timeout
     */
    int setShutter( bool shut /**< [in] true to shut */ );

    /// Take the dark after the run: shut, discard, average dark.n frames, open, and save `_<sci>_dark.fits`.
    /** The shutter is always commanded open again once shut, even on error or stop.
     * \returns 0 on success, 1 if stopped, -1 on error
     */
    int takeDark();

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

inline bool xlinqCal::looksDoubled( const std::string &name )
{
    size_t h = name.size() / 2;

    return h > 0 && name.size() % 2 == 0 && name.compare( 0, h, name, h, h ) == 0;
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

inline int xlinqCal::nNonlin() const
{
    return m_nNonlin;
}

inline std::string xlinqCal::outputPrefix() const
{
    return outputPath( "" );
}

inline const std::vector<timingRecord> &xlinqCal::timing() const
{
    return m_timing;
}

inline const camProducts &xlinqCal::sciProducts() const
{
    return m_sciProd;
}

inline size_t xlinqCal::nWfs() const
{
    return m_wfs.size();
}

inline const camProducts &xlinqCal::wfsProducts( size_t j ) const
{
    return m_wfs[j]->products();
}

inline int xlinqCal::nRef() const
{
    return m_nRef;
}

inline bool xlinqCal::refWeights( const std::vector<double> &tRef, double t, size_t &i0, size_t &i1, double &w1 )
{
    i0 = 0;
    i1 = 0;
    w1 = 0;

    if( tRef.empty() )
    {
        return false;
    }

    size_t j = std::upper_bound( tRef.begin(), tRef.end(), t ) - tRef.begin();

    if( j == 0 || j == tRef.size() )
    {
        i0 = ( j == 0 ) ? 0 : tRef.size() - 1;
        i1 = i0;
        return false;
    }

    i0 = j - 1;
    i1 = j;
    w1 = ( t - tRef[i0] ) / ( tRef[i1] - tRef[i0] );

    return true;
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
                "FITS cube of DM shapes (rows x cols x Npoke), each normalized to 1 micron rms over the active "
                "actuators. Normalization is not checked. Required." );
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

    config.add( "sci.linearMax",
                "",
                "sci.linearMax",
                argType::Required,
                "sci",
                "linearMax",
                false,
                "float",
                "Science camera linearity limit [counts]. Acquisitions with a pixel above it are counted and warned; "
                "0 disables. Default 37000." );

    config.add( "wfs.shmim",
                "",
                "wfs.shmim",
                argType::Required,
                "wfs",
                "shmim",
                false,
                "vector<string>",
                "The WFS streams, comma separated, e.g. aol1_imWFS2pub,aol1_modevalWFSpub. Default camwfs." );

    config.add( "snap.shmim",
                "",
                "snap.shmim",
                argType::Required,
                "snap",
                "shmim",
                false,
                "vector<string>",
                "Streams to save at the start (and end, if changed), comma separated, e.g. "
                "aol1_wfsref,aol1_wfsmask. Default none." );

    config.add( "indi.enable",
                "",
                "indi.enable",
                argType::Required,
                "indi",
                "enable",
                false,
                "bool",
                "Record INDI state (getINDI) at start and end, and allow a dark. No INDI calls are made unless true. "
                "Default false." );
    config.add( "indi.get",
                "",
                "indi.get",
                argType::Required,
                "indi",
                "get",
                false,
                "string",
                "The getINDI command. Default getINDI." );
    config.add( "indi.set",
                "",
                "indi.set",
                argType::Required,
                "indi",
                "set",
                false,
                "string",
                "The setINDI command. Default setINDI." );
    config.add( "indi.timeout",
                "",
                "indi.timeout",
                argType::Required,
                "indi",
                "timeout",
                false,
                "float",
                "Wall-clock limit for each INDI call [s]. Default 8." );
    config.add( "indi.poll",
                "",
                "indi.poll",
                argType::Required,
                "indi",
                "poll",
                false,
                "float",
                "Interval between readbacks while waiting for a property [s]. Default 2 (the MagAOX minimum)." );
    config.add( "indi.props",
                "",
                "indi.props",
                argType::Required,
                "indi",
                "props",
                false,
                "vector<string>",
                "Extra device.property names to record, comma separated (exact names only, no wildcards). "
                "Default none." );
    config.add( "sci.device",
                "",
                "sci.device",
                argType::Required,
                "sci",
                "device",
                false,
                "string",
                "The science camera INDI device, for exposure time, EM gain, and the shutter. Default camsci1." );
    config.add( "dark.n",
                "",
                "dark.n",
                argType::Required,
                "dark",
                "n",
                false,
                "int",
                "Frames averaged for a dark taken with the shutter closed after a complete run; needs "
                "indi.enable. 0 for no dark. Default 0." );
    config.add( "dark.discard",
                "",
                "dark.discard",
                argType::Required,
                "dark",
                "discard",
                false,
                "int",
                "Frames discarded after the shutter closes. Default 2." );
    config.add( "dark.wait",
                "",
                "dark.wait",
                argType::Required,
                "dark",
                "wait",
                false,
                "float",
                "Timeout for the shutter to reach the requested state [s]. Default 15." );

    config.add( "seq.nRepeat",
                "",
                "seq.nRepeat",
                argType::Required,
                "seq",
                "nRepeat",
                false,
                "int",
                "Number of +/- pairs per poke, alternating +- and -+. Default 2." );

    config.add( "ref.every",
                "",
                "ref.every",
                argType::Required,
                "ref",
                "every",
                false,
                "int",
                "Pokes between zero-poke reference acquisitions, 0 disables references and the even part. "
                "Default 1." );

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
                "Output file prefix, followed by _<UTC timestamp> unless out.timestamp=false. Default xlinqCal." );
    config.add(
        "out.timestamp",
        "",
        "out.timestamp",
        argType::Required,
        "out",
        "timestamp",
        false,
        "bool",
        "Append _<UTC timestamp> to the prefix. Default true. A prefix already used in out.dir is always refused." );
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
    config( m_sciLinearMax, "sci.linearMax" );

    config( m_wfsNames, "wfs.shmim" );
    config( m_snapNames, "snap.shmim" );
    config( m_indiEnable, "indi.enable" );
    config( m_indiGet, "indi.get" );
    config( m_indiSet, "indi.set" );
    config( m_indiTimeout, "indi.timeout" );
    config( m_indiPoll, "indi.poll" );
    config( m_indiProps, "indi.props" );
    config( m_sciDevice, "sci.device" );
    config( m_darkN, "dark.n" );
    config( m_darkDiscard, "dark.discard" );
    config( m_darkWait, "dark.wait" );

    m_indi.commands( m_indiGet, m_indiSet );
    m_indi.timeout( m_indiTimeout );

    config( m_nRepeat, "seq.nRepeat" );

    config( m_refEvery, "ref.every" );

    config( m_outDir, "out.dir" );
    config( m_outPrefix, "out.prefix" );
    config( m_outTimestamp, "out.timestamp" );
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

    if( resolvePrefix() < 0 || loadPokes() < 0 || openStreams() < 0 )
    {
        return -1;
    }

    if( m_darkN > 0 && !m_indiEnable )
    {
        logError( "dark.n > 0 needs indi.enable=true" );
        return -1;
    }

    int rv = preflight();
    if( rv != 0 )
    {
        return rv;
    }

    if( indiPreflight() < 0 )
    {
        return -1;
    }

    if( setupOutput() < 0 || savePokes() < 0 || saveStartSnapshots() < 0 )
    {
        return -1;
    }

    if( m_indiEnable && indiRecord( "_indi.txt", m_indiStart ) < 0 )
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

    if( checkEndSnapshots() < 0 )
    {
        rv = -1;
    }

    if( m_darkN > 0 && rv == 0 )
    {
        int drv = takeDark();

        if( drv != 0 )
        {
            logError( "no dark saved" );
            rv = drv;
        }
    }
    else if( m_darkN > 0 )
    {
        log( "run did not complete: no dark taken" );
    }

    if( m_indiEnable )
    {
        std::map<std::string, std::string> endVals;
        indiRecord( "_indi_end.txt", endVals );

        for( const char *e : { ".exptime.current", ".emgain.current" } )
        {
            std::string key = m_sciDevice + e;
            auto        a   = m_indiStart.find( key );
            auto        b   = endVals.find( key );

            if( a != m_indiStart.end() && b != endVals.end() && a->second != b->second )
            {
                log( "warning: " + key + " changed during the run: " + a->second + " -> " + b->second );
            }
        }
    }

    clock_gettime( CLOCK_REALTIME, &ts );
    m_dateEnd = isoTime( ts );

    if( writeOutput( true ) < 0 )
    {
        rv = -1;
    }

    if( m_nNonlin > 0 )
    {
        log( "warning: " + std::to_string( m_nNonlin ) + " acquisitions had " + m_sciName +
             " pixels above sci.linearMax=" + std::to_string( m_sciLinearMax ) + " (see the SCIMAX timing column)" );
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
    if( ff.read( m_pokes, m_pokeHead, m_pokeFile ) != mx::error_t::noerror )
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

    return 0;
}

inline int xlinqCal::openStreams()
{
    std::vector<std::string> names = { m_dmChannel, m_dmComb, m_sciName };
    names.insert( names.end(), m_wfsNames.begin(), m_wfsNames.end() );
    names.insert( names.end(), m_snapNames.begin(), m_snapNames.end() );

    for( const std::string &n : names )
    {
        if( looksDoubled( n ) )
        {
            logError( "stream name " + n +
                      " looks doubled: is the key repeated in the config file? (repeated keys are joined)" );
            return -1;
        }
    }

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

    if( m_wfsNames.empty() )
    {
        logError( "wfs.shmim must name at least one stream" );
        return -1;
    }

    try
    {
        m_comb.open( m_dmComb );
        m_sci.open( m_sciName );

        m_wfs.clear();
        for( const std::string &name : m_wfsNames )
        {
            m_wfs.push_back( std::make_unique<wfsChannel>() );
            m_wfs.back()->open( name );
        }

        m_snap.clear();
        for( const std::string &name : m_snapNames )
        {
            m_snap.push_back( std::make_unique<isioStream>() );
            m_snap.back()->open( name, false );
        }
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

    int nPoke = m_pokeLast - m_pokeFirst + 1;
    int nRef  = ( m_refEvery > 0 ) ? ( nPoke + m_refEvery - 1 ) / m_refEvery + 1 : 0;

    m_sciProd.allocate( m_sci.rows(), m_sci.cols(), nPoke, nRef );

    for( auto &w : m_wfs )
    {
        w->products().allocate( w->stream().rows(), w->stream().cols(), nPoke, nRef );
    }

    return 0;
}

inline int xlinqCal::preflight()
{
    uint64_t              sci0 = m_sci.cnt0();
    std::vector<uint64_t> wfs0;

    for( auto &w : m_wfs )
    {
        wfs0.push_back( w->stream().cnt0() );
    }

    double t0 = nowSec();

    std::this_thread::sleep_for( std::chrono::duration<double>( m_preflightTime ) );

    double dt = nowSec() - t0;
    m_sciFps  = ( m_sci.cnt0() - sci0 ) / dt;

    if( m_sciFps <= 0 )
    {
        logError( m_sciName + " is not updating (check camera and shmimTCP)" );
        return -1;
    }

    double wfsPix = 0;

    for( size_t j = 0; j < m_wfs.size(); ++j )
    {
        m_wfs[j]->fps( ( m_wfs[j]->stream().cnt0() - wfs0[j] ) / dt );

        if( m_wfs[j]->fps() <= 0 )
        {
            logError( m_wfsNames[j] + " is not updating" );
            return -1;
        }

        wfsPix += m_wfs[j]->stream().rows() * m_wfs[j]->stream().cols();
    }

    {
        mx::improc::eigenImage<double> frame;
        uint64_t                       cnt0;
        m_sci.copy( frame, cnt0 );
        m_sciPreMax = frame.maxCoeff();
    }

    std::vector<int> signs         = signSequence( m_nRepeat );
    int              nPoke         = m_pokeLast - m_pokeFirst + 1;
    double           framesPerPoke = settleCount( signs ) * ( 1.0 + m_sciSettle ) + signs.size() * m_sciFrames;
    int              nRef          = ( m_refEvery > 0 ) ? ( nPoke + m_refEvery - 1 ) / m_refEvery + 1 : 0;
    double           frames        = nPoke * framesPerPoke + nRef * ( 1.0 + m_sciSettle + m_sciFrames );
    double           hours         = frames / m_sciFps / 3600.;
    double           planes        = ( m_refEvery > 0 ) ? 3.0 * nPoke + nRef : 2.0 * nPoke; // resp, mean, [even, ref]
    double           gbytes        = planes * 4.0 * ( m_sci.rows() * m_sci.cols() + wfsPix ) / 1e9;
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
       << " at " << m_sciFps << " fps, max pixel " << m_sciPreMax;
    if( m_sciLinearMax > 0 )
    {
        os << " (linear limit " << m_sciLinearMax << ( m_sciPreMax > m_sciLinearMax ? ", EXCEEDED" : "" ) << ")";
    }
    os << "\n";
    for( size_t j = 0; j < m_wfs.size(); ++j )
    {
        const isioStream &ws = m_wfs[j]->stream();
        os << "  wfs " << j << ":   " << m_wfsNames[j] << " " << ws.rows() << "x" << ws.cols() << " type "
           << (int)ws.datatype() << " at " << m_wfs[j]->fps() << " fps (~" << m_wfs[j]->fps() / m_sciFps
           << " per sci frame)\n";
    }
    os << "  timing:  discard 1+" << m_sciSettle << ", average " << m_sciFrames << " sci frames, sequence " << seq
       << ", reference every " << m_refEvery << " pokes (" << nRef << " total)\n";
    if( m_indiEnable )
    {
        os << "  indi:    record " << indiPropList().size() << " properties at start and end; ";
        os << ( m_darkN > 0 ? "dark of " + std::to_string( m_darkN ) + " frames after the run (" + m_sciDevice +
                                  " shutter shut, then opened)"
                            : std::string( "no dark" ) )
           << "\n";
    }
    os << "  output:  " << outputPath( "_*" ) << "\n";
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

inline int xlinqCal::resolvePrefix()
{
    if( m_outPrefix.empty() )
    {
        m_outPrefix = "xlinqCal";
    }

    if( m_outTimestamp )
    {
        timespec ts;
        clock_gettime( CLOCK_REALTIME, &ts );

        tm uttime;
        gmtime_r( &ts.tv_sec, &uttime );

        char buf[32];
        strftime( buf, sizeof( buf ), "_%Y%m%dT%H%M%S", &uttime );
        m_outPrefix += buf;
    }

    std::string example;
    if( prefixInUse( example ) )
    {
        logError( "output prefix " + outputPath( "" ) + " is already used (" + example +
                  "); choose another out.prefix so earlier results are not overwritten" );
        return -1;
    }

    return 0;
}

inline bool xlinqCal::prefixInUse( std::string &example ) const
{
    std::error_code ec;

    for( const auto &entry : std::filesystem::directory_iterator( m_outDir, ec ) )
    {
        std::string name = entry.path().filename().string();

        if( name.size() > m_outPrefix.size() && name.compare( 0, m_outPrefix.size(), m_outPrefix ) == 0 &&
            ( name[m_outPrefix.size()] == '_' || name[m_outPrefix.size()] == '.' ) )
        {
            example = name;
            return true;
        }
    }

    return false;
}

inline int xlinqCal::setupOutput()
{
    std::string example;
    if( prefixInUse( example ) )
    {
        logError( "output prefix " + outputPath( "" ) + " was used during the preflight (" + example + ")" );
        return -1;
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

inline void xlinqCal::startWfsThread()
{
    for( auto &w : m_wfs )
    {
        w->start();
    }
}

inline void xlinqCal::stopWfsThread()
{
    for( auto &w : m_wfs )
    {
        w->stop();
    }
}

inline timingRecord xlinqCal::newRecord() const
{
    return timingRecord( timingCol::nCommon + m_wfs.size() * wfsCol::nCols, std::numeric_limits<double>::quiet_NaN() );
}

inline int xlinqCal::run()
{
    int    nPoke    = m_pokeLast - m_pokeFirst + 1;
    double tStart   = nowSec();
    double tLastChk = tStart;
    int    rv;

    m_sciLastCnt0 = m_sci.cnt0();

    if( m_refEvery > 0 && ( rv = measureRef() ) != 0 )
    {
        return rv;
    }

    for( int k = m_pokeFirst; k <= m_pokeLast; ++k )
    {
        if( s_stop )
        {
            return 1;
        }

        if( ( rv = measurePoke( k ) ) != 0 )
        {
            return rv;
        }

        ++m_nDone;

        if( m_refEvery > 0 && ( m_nDone % m_refEvery == 0 || k == m_pokeLast ) && ( rv = measureRef() ) != 0 )
        {
            return rv;
        }

        const timingRecord &last    = m_timing.back();
        double              elapsed = nowSec() - tStart;
        double              eta     = elapsed / m_nDone * ( nPoke - m_nDone );

        std::ostringstream os;
        os << "poke " << k << " (" << m_nDone << "/" << nPoke << ") wfs frames/window "
           << last[wfsColIndex( 0, wfsCol::n )] << ", elapsed " << elapsed / 60 << " min, remaining " << eta / 60
           << " min";
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
    size_t           nW    = m_wfs.size();

    mx::improc::eigenImage<double>              sciMean;
    std::vector<mx::improc::eigenImage<double>> wfsMeans( nW );

    // Signed and symmetric accumulators: index 0 is camsci1, 1..nW are the WFS streams
    std::vector<mx::improc::eigenImage<double>> odd( nW + 1 ), sym( nW + 1 );

    for( size_t j = 0; j <= nW; ++j )
    {
        const isioStream &st = ( j == 0 ) ? m_sci : m_wfs[j - 1]->stream();
        odd[j].setZero( st.rows(), st.cols() );
        sym[j].setZero( st.rows(), st.cols() );
    }

    double tCenter = 0;

    for( size_t i = 0; i < signs.size(); ++i )
    {
        timingRecord rec       = newRecord();
        rec[timingCol::poke]   = k;
        rec[timingCol::repeat] = i / 2;
        rec[timingCol::sign]   = signs[i];

        mx::improc::eigenImage<float> cmd = m_dmInitial + ( signs[i] * m_pokeAmp ) * m_pokes.image( k );

        int rv = setAndAcquire( cmd, sciMean, wfsMeans, rec );
        if( rv != 0 )
        {
            return rv;
        }

        for( size_t j = 0; j <= nW; ++j )
        {
            const mx::improc::eigenImage<double> &mean = ( j == 0 ) ? sciMean : wfsMeans[j - 1];
            odd[j] += static_cast<double>( signs[i] ) * mean;
            sym[j] += mean;
        }

        tCenter += 0.5 * ( rec[timingCol::tStart] + rec[timingCol::tEnd] );

        m_timing.push_back( rec );
    }

    // (mean(+) - mean(-))/2 and (mean(+) + mean(-))/2, averaged over repeats
    double nAcq = signs.size();

    for( size_t j = 0; j <= nW; ++j )
    {
        camProducts &prod                  = ( j == 0 ) ? m_sciProd : m_wfs[j - 1]->products();
        prod.resp.image( k - m_pokeFirst ) = ( odd[j] / nAcq ).cast<float>();
        prod.mean.image( k - m_pokeFirst ) = ( sym[j] / nAcq ).cast<float>();
    }

    m_pokeTime.push_back( tCenter / nAcq );

    return 0;
}

inline int xlinqCal::measureRef()
{
    timingRecord rec       = newRecord();
    rec[timingCol::repeat] = m_nRef;
    rec[timingCol::sign]   = 0;

    mx::improc::eigenImage<double>              sciMean;
    std::vector<mx::improc::eigenImage<double>> wfsMeans( m_wfs.size() );

    int rv = setAndAcquire( m_dmInitial, sciMean, wfsMeans, rec );
    if( rv != 0 )
    {
        return rv;
    }

    m_sciProd.ref.image( m_nRef ) = sciMean.cast<float>();

    for( size_t j = 0; j < m_wfs.size(); ++j )
    {
        m_wfs[j]->products().ref.image( m_nRef ) = wfsMeans[j].cast<float>();
    }

    m_refTime.push_back( 0.5 * ( rec[timingCol::tStart] + rec[timingCol::tEnd] ) );
    ++m_nRef;

    m_timing.push_back( rec );

    return 0;
}

inline int xlinqCal::setAndAcquire( const mx::improc::eigenImage<float>         &cmd,
                                    mx::improc::eigenImage<double>              &sciMean,
                                    std::vector<mx::improc::eigenImage<double>> &wfsMeans,
                                    timingRecord                                &rec )
{
    bool changed = !( cmd == m_dmLast ).all();

    if( changed )
    {
        int rv = applyCommand( cmd, rec );
        if( rv != 0 )
        {
            return rv;
        }
    }

    bool settle = changed || m_firstAcq;
    m_firstAcq  = false;

    return acquire( settle, sciMean, wfsMeans, rec );
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

inline int xlinqCal::waitSciFrame( mx::improc::eigenImage<double> *frame, uint64_t &cnt0, timespec *atime )
{
    double tWait = nowSec() + m_sciWait;

    while( true )
    {
        if( s_stop )
        {
            return 1;
        }

        for( size_t j = 0; j < m_wfs.size(); ++j )
        {
            if( m_wfs[j]->error() != 0 )
            {
                logError( "WFS thread error waiting on " + m_wfsNames[j] );
                return -1;
            }
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
        else
        {
            timespec at;

            if( !m_sci.copy( *frame, cnt0, at ) )
            {
                continue; // Torn: wait for the next frame, the gap is counted as missed
            }

            if( atime != nullptr )
            {
                *atime = at;
            }
        }

        m_sciLastCnt0 = cnt0;

        return 0;
    }
}

inline int xlinqCal::acquire( bool                                         settle,
                              mx::improc::eigenImage<double>              &sciMean,
                              std::vector<mx::improc::eigenImage<double>> &wfsMeans,
                              timingRecord                                &rec )
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

    // Open the gates: the next camsci1 exposure starts now.
    for( auto &w : m_wfs )
    {
        w->openGate();
    }

    rec[timingCol::tStart] = nowSec();

    mx::improc::eigenImage<double> frame;
    mx::improc::eigenImage<double> sciSum( m_sci.rows(), m_sci.cols() );
    sciSum.setZero();

    uint64_t sciMissed = 0;
    uint64_t prev      = m_sciLastCnt0;
    timespec atime{};
    double   sciMax = -std::numeric_limits<double>::infinity();

    for( unsigned n = 0; n < m_sciFrames; ++n )
    {
        if( ( rv = waitSciFrame( &frame, cnt0, &atime ) ) != 0 )
        {
            wfsGateStats st;
            for( size_t j = 0; j < m_wfs.size(); ++j )
            {
                m_wfs[j]->closeGate( wfsMeans[j], st );
            }

            return rv;
        }

        if( n == 0 )
        {
            rec[timingCol::sciCnt0First]  = cnt0;
            rec[timingCol::sciAtimeFirst] = tsSec( atime );
        }
        else if( cnt0 > prev + 1 )
        {
            sciMissed += cnt0 - prev - 1;
        }

        prev = cnt0;
        sciSum += frame;
        sciMax = std::max( sciMax, frame.maxCoeff() );
    }

    rec[timingCol::sciAtimeLast] = tsSec( atime );
    rec[timingCol::sciMax]       = sciMax;

    if( m_sciLinearMax > 0 && sciMax > m_sciLinearMax )
    {
        if( m_nNonlin == 0 )
        {
            log( "warning: " + m_sciName + " pixel value " + std::to_string( sciMax ) + " is above sci.linearMax=" +
                 std::to_string( m_sciLinearMax ) + "; further occurrences are counted, not logged" );
        }

        ++m_nNonlin;
    }

    bool allFrames = true;

    for( size_t j = 0; j < m_wfs.size(); ++j )
    {
        wfsGateStats st;

        if( !m_wfs[j]->closeGate( wfsMeans[j], st ) )
        {
            logError( "no " + m_wfsNames[j] + " frames received during the averaging window" );
            allFrames = false;
        }

        rec[wfsColIndex( j, wfsCol::cnt0First )]  = st.cnt0First;
        rec[wfsColIndex( j, wfsCol::cnt0Last )]   = st.cnt0Last;
        rec[wfsColIndex( j, wfsCol::n )]          = st.n;
        rec[wfsColIndex( j, wfsCol::missed )]     = st.missed;
        rec[wfsColIndex( j, wfsCol::torn )]       = st.torn;
        rec[wfsColIndex( j, wfsCol::atimeFirst )] = st.atimeFirst;
        rec[wfsColIndex( j, wfsCol::atimeLast )]  = st.atimeLast;
    }

    rec[timingCol::tEnd]        = nowSec();
    rec[timingCol::sciCnt0Last] = prev;
    rec[timingCol::sciN]        = m_sciFrames;
    rec[timingCol::sciMissed]   = sciMissed;

    if( !allFrames )
    {
        return -1;
    }

    sciMean = sciSum / static_cast<double>( m_sciFrames );

    return 0;
}

inline int xlinqCal::readSnapshots( std::vector<mx::improc::eigenImage<float>> &ims )
{
    ims.resize( m_snap.size() );

    for( size_t j = 0; j < m_snap.size(); ++j )
    {
        uint64_t cnt0;
        int      tries = 0;

        while( !m_snap[j]->copy( ims[j], cnt0 ) )
        {
            if( ++tries > 100 )
            {
                logError( "snapshot of " + m_snapNames[j] + " kept changing during the copy" );
                return -1;
            }
        }
    }

    return 0;
}

inline int xlinqCal::writeSnapshot( size_t j, const mx::improc::eigenImage<float> &im, const std::string &suffix )
{
    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );

    timespec wt = m_snap[j]->writetime();

    mx::fits::fitsHeader<> head;
    head.append( "XLQVERS", std::string( XLINQCAL_GIT_VERSION ), "xlinqCal version" );
    head.append( "SNAPSTRM", m_snapNames[j], "snapshot stream" );
    head.append( "SNAPCNT0", (long)m_snap[j]->cnt0(), "stream cnt0 at snapshot" );
    head.append( "SNAPWRT", isoTime( wt ), "stream write time, UTC" );
    head.append( "DATE-OBS", isoTime( ts ), "snapshot time, UTC" );
    head.append( "SNAPTYPE", (int)m_snap[j]->datatype(), "stream ImageStreamIO datatype (saved as float)" );

    std::string               file = outputPath( "_snap_" + m_snapNames[j] + suffix + ".fits" );
    mx::fits::fitsFile<float> ff;

    if( ff.write( file, im, head ) != mx::error_t::noerror )
    {
        logError( "could not write " + file );
        return -1;
    }

    return 0;
}

inline int xlinqCal::savePokes()
{
    int nPoke = m_pokeLast - m_pokeFirst + 1;

    mx::improc::eigenCube<float> applied( m_pokes.rows(), m_pokes.cols(), nPoke );

    for( int k = 0; k < nPoke; ++k )
    {
        applied.image( k ) = m_pokeAmp * m_pokes.image( m_pokeFirst + k );
    }

    mx::fits::fitsHeader<> head = m_pokeHead;
    head.eraseStandardTop();
    head.append( "XLQVERS", std::string( XLINQCAL_GIT_VERSION ), "xlinqCal version" );
    head.append( "POKEFILE", m_pokeFile, "source poke file; its header cards precede these" );
    head.append( "POKEAMP", m_pokeAmp, "multiplier applied to the poke file" );
    head.append( "POKE0", m_pokeFirst, "poke file index of first plane" );
    head.append( "NPOKEREQ", nPoke, "number of pokes requested" );
    head.append( "PRODUCT", std::string( "applied pokes: POKEAMP * poke file planes [um]" ), "content of this cube" );

    std::string               file = outputPath( "_pokes.fits" );
    mx::fits::fitsFile<float> ff;

    if( ff.write( file, applied, head ) != mx::error_t::noerror )
    {
        logError( "could not write " + file );
        return -1;
    }

    return 0;
}

inline int xlinqCal::saveStartSnapshots()
{
    if( readSnapshots( m_snapStart ) < 0 )
    {
        return -1;
    }

    for( size_t j = 0; j < m_snap.size(); ++j )
    {
        if( writeSnapshot( j, m_snapStart[j], "" ) < 0 )
        {
            return -1;
        }

        log( "saved snapshot of " + m_snapNames[j] );
    }

    return 0;
}

inline int xlinqCal::checkEndSnapshots()
{
    std::vector<mx::improc::eigenImage<float>> end;

    if( readSnapshots( end ) < 0 )
    {
        return -1;
    }

    for( size_t j = 0; j < m_snap.size(); ++j )
    {
        bool same = end[j].rows() == m_snapStart[j].rows() && end[j].cols() == m_snapStart[j].cols() &&
                    ( end[j] == m_snapStart[j] ).all();

        if( !same )
        {
            log( "warning: " + m_snapNames[j] + " changed during the run; end state saved with suffix _end" );

            if( writeSnapshot( j, end[j], "_end" ) < 0 )
            {
                return -1;
            }
        }
    }

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

inline std::vector<std::string> xlinqCal::indiPropList() const
{
    std::vector<std::string> props;

    for( const char *p : { ".fsm", ".exptime", ".emgain", ".fps", ".shutter_status", ".shutter" } )
    {
        props.push_back( m_sciDevice + p );
    }

    for( const std::string &p : m_indiProps )
    {
        if( std::find( props.begin(), props.end(), p ) == props.end() )
        {
            props.push_back( p );
        }
    }

    return props;
}

inline int xlinqCal::indiPreflight()
{
    if( !m_indiEnable )
    {
        return 0;
    }

    if( !indiClient::validName( m_sciDevice + ".exptime", 1 ) )
    {
        logError( "invalid sci.device '" + m_sciDevice + "'" );
        return -1;
    }

    for( const std::string &p : m_indiProps )
    {
        if( !indiClient::validName( p, 1 ) )
        {
            logError( "invalid indi.props entry '" + p + "' (expected device.property, no wildcards)" );
            return -1;
        }
    }

    std::string val, err;

    if( m_indi.getElement( m_sciDevice + ".exptime.current", val, err ) < 0 )
    {
        logError( "INDI check failed: " + err );
        return -1;
    }

    log( "INDI: " + m_sciDevice + " exposure time " + val + " s" );

    if( m_darkN == 0 )
    {
        return 0;
    }

    if( m_indi.getElement( m_sciDevice + ".shutter_status.status", val, err ) < 0 )
    {
        logError( "INDI shutter check failed: " + err );
        return -1;
    }

    if( val != "READY" )
    {
        logError( m_sciDevice + " shutter status is " + val + ", not READY: can't take a dark" );
        return -1;
    }

    if( m_indi.getElement( m_sciDevice + ".shutter.toggle", val, err ) < 0 )
    {
        logError( "INDI shutter check failed: " + err );
        return -1;
    }

    if( val != "Off" )
    {
        logError( m_sciDevice + " shutter is shut (shutter.toggle=" + val + "): open it before calibrating" );
        return -1;
    }

    return 0;
}

inline int xlinqCal::indiRecord( const std::string &suffix, std::map<std::string, std::string> &vals )
{
    vals.clear();

    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );

    std::string   file = outputPath( suffix );
    std::ofstream fout( file );

    if( !fout )
    {
        logError( "could not write " + file );
        return -1;
    }

    fout << "# xlinqCal " << XLINQCAL_GIT_VERSION << " INDI state at " << isoTime( ts ) << " UTC\n";

    int nBad = 0;

    for( const std::string &p : indiPropList() )
    {
        std::map<std::string, std::string> elems;
        std::string                        raw, err;

        if( m_indi.getProperty( p, elems, raw, err ) < 0 )
        {
            fout << "# " << p << ": " << err << "\n";
            log( "warning: INDI " + err );
            ++nBad;
            continue;
        }

        fout << raw;

        for( const auto &e : elems )
        {
            vals[p + "." + e.first] = e.second;
        }
    }

    if( suffix == "_indi.txt" )
    {
        auto num = [&]( const std::string &key )
        {
            auto it = vals.find( m_sciDevice + key );
            return it == vals.end() ? std::numeric_limits<double>::quiet_NaN() : atof( it->second.c_str() );
        };

        m_sciExptime = num( ".exptime.current" );
        m_sciEmgain  = num( ".emgain.current" );
    }

    log( "recorded INDI state to " + file + ( nBad > 0 ? " (" + std::to_string( nBad ) + " unreadable)" : "" ) );

    return 0;
}

inline int xlinqCal::setShutter( bool shut )
{
    std::string elem = m_sciDevice + ".shutter.toggle";
    std::string want = shut ? "On" : "Off";
    std::string val, err;

    if( m_indi.setSwitch( elem, shut, err ) < 0 )
    {
        logError( "shutter command failed: " + err );
        return -1;
    }

    double tEnd = nowSec() + m_darkWait;

    while( true )
    {
        std::this_thread::sleep_for( std::chrono::duration<double>( m_indiPoll ) );

        if( m_indi.getElement( elem, val, err ) == 0 && val == want )
        {
            log( m_sciDevice + " shutter " + ( shut ? "shut" : "open" ) );
            return 0;
        }

        if( nowSec() > tEnd )
        {
            logError( m_sciDevice + " shutter did not " + ( shut ? "shut" : "open" ) + " within dark.wait (" + elem +
                      "=" + ( err.empty() ? val : err ) + ")" );
            return -1;
        }
    }
}

inline int xlinqCal::takeDark()
{
    log( "taking dark: " + std::to_string( m_darkN ) + " frames" );

    if( setShutter( true ) < 0 )
    {
        // It may have shut late: always command it open again
        m_shutterShut = true;

        if( setShutter( false ) == 0 )
        {
            m_shutterShut = false;
        }
        else
        {
            logError( "THE " + m_sciDevice + " SHUTTER MAY BE SHUT: check and open it" );
        }

        return -1;
    }

    m_shutterShut = true;

    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    std::string darkBeg = isoTime( ts );

    int                            rv = 0;
    uint64_t                       cnt0;
    mx::improc::eigenImage<double> frame;
    mx::improc::eigenImage<double> sum( m_sci.rows(), m_sci.cols() );
    sum.setZero();

    m_sci.flush();
    m_sciLastCnt0 = m_sci.cnt0();

    for( unsigned n = 0; n < m_darkDiscard && rv == 0; ++n )
    {
        rv = waitSciFrame( nullptr, cnt0 );
    }

    for( unsigned n = 0; n < m_darkN && rv == 0; ++n )
    {
        if( ( rv = waitSciFrame( &frame, cnt0 ) ) == 0 )
        {
            sum += frame;
        }
    }

    clock_gettime( CLOCK_REALTIME, &ts );
    std::string darkEnd = isoTime( ts );

    // Always open the shutter again, even if stopped or on error
    if( setShutter( false ) == 0 )
    {
        m_shutterShut = false;
    }
    else
    {
        logError( "THE " + m_sciDevice + " SHUTTER MAY STILL BE SHUT: check and open it" );
        rv = -1;
    }

    if( rv != 0 )
    {
        return rv;
    }

    mx::improc::eigenImage<float> dark = ( sum / static_cast<double>( m_darkN ) ).cast<float>();

    if( m_sciPreMax > 0 && dark.maxCoeff() > 0.5 * m_sciPreMax )
    {
        log( "warning: dark max " + std::to_string( dark.maxCoeff() ) + " is not far below the preflight max " +
             std::to_string( m_sciPreMax ) + ": did the shutter close?" );
    }

    m_darkTaken = m_darkN;

    mx::fits::fitsHeader<> head = outputHeader( true );
    head.append( "PRODUCT", std::string( "dark: mean of NDARK frames with the shutter shut" ), "content of this cube" );
    head.append( "DARKBEG", darkBeg, "UTC shutter confirmed shut" );
    head.append( "DARKEND", darkEnd, "UTC last dark frame" );
    head.append( "DARKDISC", m_darkDiscard, "frames discarded after the shutter shut" );

    std::string               file = outputPath( "_" + m_sciName + "_dark.fits" );
    mx::fits::fitsFile<float> ff;

    if( ff.write( file, dark, head ) != mx::error_t::noerror )
    {
        logError( "could not write " + file );
        m_darkTaken = 0;
        return -1;
    }

    log( "wrote dark to " + file );

    return 0;
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
    head.append( "POKEFILE", m_pokeFile, "poke shapes, 1 um rms each" );
    head.append( "POKEAMP", m_pokeAmp, "poke amplitude [um rms]" );
    head.append( "POKE0", m_pokeFirst, "poke index of first plane" );
    head.append( "NPOKEREQ", m_pokeLast - m_pokeFirst + 1, "number of pokes requested" );
    head.append( "NPOKE", m_nDone, "number of planes measured" );
    head.append( "NSETTLE", m_sciSettle, "sci frames discarded after the first post-poke frame" );
    head.append( "NFRAMES", m_sciFrames, "sci frames averaged per sign" );
    head.append( "NREPEAT", m_nRepeat, "+/- pairs per poke" );
    head.append( "SEQUENCE", seq, "sign sequence per poke" );
    head.append( "REFEVERY", m_refEvery, "pokes between zero-poke references, 0 = none" );
    head.append( "NREF", m_nRef, "number of references measured" );
    head.append( "DMCHAN", m_dmChannel, "DM channel poked" );
    head.append( "DMCOMB", m_dmComb, "combined DM stream" );
    head.append( "SCISTRM", m_sciName, "science camera stream" );
    head.append( "SCIFPS", m_sciFps, "measured sci frame rate [Hz]" );
    head.append( "SCILINMX", m_sciLinearMax, "sci linearity limit [counts], 0 = off" );
    head.append( "SCIPRMAX", m_sciPreMax, "sci max pixel in a preflight frame" );
    head.append( "NNONLIN", m_nNonlin, "acquisitions with sci pixels above SCILINMX" );
    head.append( "SCITYPE", (int)m_sci.datatype(), "sci ImageStreamIO datatype" );
    head.append( "INDI", m_indiEnable, "INDI state recorded, see _indi*.txt" );
    head.append( "SCIDEV", m_sciDevice, "sci camera INDI device" );
    if( std::isfinite( m_sciExptime ) ) // NaN cards can't be written
    {
        head.append( "SCIEXPT", m_sciExptime, "sci exposure time from INDI at start [s]" );
    }
    if( std::isfinite( m_sciEmgain ) )
    {
        head.append( "SCIEMGN", m_sciEmgain, "sci EM gain from INDI at start" );
    }
    head.append( "NDARK", m_darkTaken, "frames in the dark (_dark.fits), 0 = none" );
    head.append( "NSNAP", (int)m_snapNames.size(), "number of snapshot streams" );

    for( size_t j = 0; j < m_snapNames.size(); ++j )
    {
        head.append( "SNAP" + std::to_string( j ), m_snapNames[j], "snapshot stream, see _snap_ files" );
    }

    head.append( "NWFS", (int)m_wfs.size(), "number of WFS streams" );

    for( size_t j = 0; j < m_wfs.size(); ++j )
    {
        std::string jj = std::to_string( j );
        head.append( "WFS" + jj + "STRM", m_wfsNames[j], "WFS stream " + jj );
        head.append( "WFS" + jj + "FPS", m_wfs[j]->fps(), "measured frame rate [Hz]" );
        head.append( "WFS" + jj + "TYPE", (int)m_wfs[j]->stream().datatype(), "ImageStreamIO datatype" );
    }
    head.append( "DATE-BEG", m_dateBeg, "UTC start of measurement" );
    head.append( "DATE-END", m_dateEnd.empty() ? std::string( "running" ) : m_dateEnd, "UTC end of measurement" );

    return head;
}

inline std::string xlinqCal::outputPath( const std::string &suffix ) const
{
    return ( std::filesystem::path( m_outDir ) / ( m_outPrefix + suffix ) ).string();
}

inline int xlinqCal::evenCube( mx::improc::eigenCube<float>       &even,
                               const mx::improc::eigenCube<float> &mean,
                               const mx::improc::eigenCube<float> &ref ) const
{
    even.resize( mean.rows(), mean.cols(), std::max( m_nDone, 1 ) );

    std::vector<double> tRef( m_refTime.begin(), m_refTime.begin() + m_nRef );

    int nOneSided = 0;

    for( int k = 0; k < m_nDone; ++k )
    {
        size_t i0, i1;
        double w1;

        if( !refWeights( tRef, m_pokeTime[k], i0, i1, w1 ) )
        {
            ++nOneSided;
        }

        even.image( k ) = mean.image( k ) - ( ( 1 - w1 ) * ref.image( i0 ) + w1 * ref.image( i1 ) ).cast<float>();
    }

    return nOneSided;
}

inline int xlinqCal::writeCube( const std::string                  &suffix,
                                const mx::improc::eigenCube<float> &cube,
                                int                                 nPlanes,
                                mx::fits::fitsHeader<>              head,
                                const std::string                  &product )
{
    head.append( "PRODUCT", product, "content of this cube" );

    std::string               file = outputPath( suffix );
    mx::fits::fitsFile<float> ff;

    // Planes are contiguous, so the first nPlanes planes are written directly.
    if( ff.write( file, cube.data(), cube.rows(), cube.cols(), nPlanes, head ) != mx::error_t::noerror )
    {
        logError( "could not write " + file );
        return -1;
    }

    return 0;
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

    auto writeCam = [&]( const std::string &name, const camProducts &prod )
    {
        const mx::improc::eigenCube<float> &resp = prod.resp;
        const mx::improc::eigenCube<float> &mean = prod.mean;
        const mx::improc::eigenCube<float> &ref  = prod.ref;

        std::string pre = "_" + name;

        rv |= writeCube( pre + "_resp.fits", resp, m_nDone, head, "(mean(+)-mean(-))/2, not divided by POKEAMP" );
        rv |= writeCube( pre + "_mean.fits", mean, m_nDone, head, "(mean(+)+mean(-))/2" );

        if( m_nRef > 0 )
        {
            mx::improc::eigenCube<float> even;
            int                          nOneSided = evenCube( even, mean, ref );

            mx::fits::fitsHeader<> evenHead = head;
            evenHead.append( "EVEN1SID", nOneSided, "planes using a one-sided (nearest) reference" );

            rv |= writeCube( pre + "_even.fits", even, m_nDone, evenHead, "mean - ref interpolated to poke time" );
            rv |= writeCube( pre + "_ref.fits", ref, m_nRef, head, "zero-poke reference" );
        }
    };

    writeCam( m_sciName, m_sciProd );

    for( size_t j = 0; j < m_wfs.size(); ++j )
    {
        writeCam( m_wfsNames[j], m_wfs[j]->products() );
    }

    std::vector<std::string> colNames = { "POKE",
                                          "REPEAT",
                                          "SIGN",
                                          "DMCNT0",
                                          "DMPROJ",
                                          "SCICNT0A",
                                          "SCICNT0B",
                                          "SCIN",
                                          "SCIMISS",
                                          "TSTART",
                                          "TEND",
                                          "SCIATIMA",
                                          "SCIATIMB",
                                          "SCIMAX" };

    for( size_t j = 0; j < m_wfs.size(); ++j )
    {
        for( const char *f : { "CNT0A", "CNT0B", "N", "MISS", "TORN", "ATIMA", "ATIMB" } )
        {
            colNames.push_back( "W" + std::to_string( j ) + f );
        }
    }

    mx::improc::eigenImage<double> table( colNames.size(), m_timing.size() );

    for( size_t r = 0; r < m_timing.size(); ++r )
    {
        for( size_t c = 0; c < colNames.size(); ++c )
        {
            table( c, r ) = m_timing[r][c];
        }
    }

    for( size_t c = 0; c < colNames.size(); ++c )
    {
        head.append( "TCOL" + std::to_string( c ), colNames[c], "timing table column" );
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
