/** \file xlinqSim.hpp
 * \brief A shared memory simulator of the DM, combined DM, camwfs, and camsci1 streams for testing xlinqCal.
 *
 * \ingroup xlinqCal_files
 */

#ifndef xlinqSim_hpp
#define xlinqSim_hpp

#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>

#include <ImageStreamIO/ImageStreamIO.h>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

namespace xlinqCal
{

/// A shared memory stream created and destroyed by the simulator.
class simStream
{
  protected:
    IMAGE m_image{}; ///< The ImageStreamIO image structure.

    bool m_open{ false }; ///< True if the stream has been created.

  public:
    /// Default c'tor.
    simStream() = default;

    /// Copying is not allowed since the stream is owned.
    simStream( const simStream & ) = delete;

    /// Copying is not allowed since the stream is owned.
    simStream &operator=( const simStream & ) = delete;

    /// D'tor, destroys the stream.
    ~simStream();

    /// Create the stream, zero filled.
    /** \throws std::runtime_error on error
     */
    void create( const std::string &name,     /**< [in] the stream name */
                 uint8_t            datatype, /**< [in] the ImageStreamIO datatype code */
                 uint32_t           rows,     /**< [in] the first dimension */
                 uint32_t           cols /**< [in] the second dimension */ );

    /// Destroy the stream, removing its files.
    void destroy();

    /// Get a pointer to the IMAGE structure.
    IMAGE *image();

    /// Get a typed pointer to the pixels.
    template <typename T>
    T *data();

    /// Mark the stream as being written.
    void setWrite();

    /// Update counters and times, and post the semaphores. atime is the current time.
    void post();

    /// Update counters and times with the given acquisition time, and post the semaphores.
    void post( const timespec &atime /**< [in] the frame acquisition time (CLOCK_REALTIME, as MagAO-X cameras) */ );
};

/// The current CLOCK_REALTIME time.
inline timespec realtimeNow()
{
    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );
    return ts;
}

/// Simulator configuration.
struct xlinqSimConfig
{
    std::string prefix{ "xlsim_" }; ///< Required prefix of all stream names, so real streams are never created.

    std::string dmChannel{ "xlsim_dm01disp04" }; ///< The simulated DM channel.

    std::string dmComb{ "xlsim_dm01disp" }; ///< The simulated combined DM stream.

    std::string wfsName{ "xlsim_camwfs" }; ///< The simulated WFS camera, uint16.

    std::string sciName{ "xlsim_camsci1" }; ///< The simulated science camera, float.

    std::string modevalName; ///< Optional modal-coefficient stream (float, nModes x 1), like aol1_modevalWFSpub. Empty
                             ///< disables.

    uint32_t nModes{ 64 }; ///< Number of simulated modal coefficients.

    double modevalNoise{ 0.001 }; ///< Per-frame noise rms of the modal coefficients [microns].

    uint32_t dmRows{ 50 }; ///< DM first dimension.

    uint32_t dmCols{ 50 }; ///< DM second dimension.

    uint32_t wfsRows{ 16 }; ///< camwfs first dimension.

    uint32_t wfsCols{ 16 }; ///< camwfs second dimension.

    uint32_t sciRows{ 12 }; ///< camsci1 first dimension.

    uint32_t sciCols{ 12 }; ///< camsci1 second dimension.

    double wfsFps{ 2000 }; ///< camwfs frame rate.

    double sciFps{ 4 }; ///< camsci1 frame rate; exposures are continuous.

    double sciSampleHz{ 1000 }; ///< Rate at which the DM state is sampled during a camsci1 exposure.

    double sciPostDelay{ 0.01 }; ///< Delay [s] from the end of a camsci1 exposure to its post (readout, transfer).

    double combDelay{ 0.0005 }; ///< Delay [s] from a channel post to the combined DM post.

    double wfsLevel{ 1000 }; ///< camwfs mean level [counts].

    double sciLevel{ 500 }; ///< camsci1 mean level [counts].

    double sciDark{ 100 }; ///< camsci1 level with the shutter shut [counts].

    std::string shutterFile; ///< If set, a file holding the shutter state: content starting "On" means shut (as
                             ///< MagAOX shutter.toggle). Read at the end of each camsci1 exposure.

    double sciCoreAmp{ 0 }; ///< Peak of a Gaussian PSF core at the camsci1 center, above sciLevel [counts]. 0 for none.

    double sciCoreSigma{ 2 }; ///< Gaussian sigma of the PSF core [px].

    double sciSat{ 0 }; ///< camsci1 saturation level: pixel values are clipped here [counts]. 0 for none.

    int sciBleed{ 0 }; ///< Saturation bleed [px]: in each column, the pixels past the last saturated one along the
                       ///< first image axis are saturated too, like camsci1's bleed stripe. 0 for none.

    std::string attenFile; ///< If set, a file holding a filter state: content starting "On" scales the camsci1
                           ///< light (everything above sciDark) by attenFactor, like an ND filter.

    double attenFactor{ 0.01 }; ///< Light transmission with the attenFile filter in.

    double wfsGain{ 50 }; ///< rms camwfs response to a 1 micron rms poke [counts].

    double sciGain{ 50 }; ///< rms camsci1 response to a 1 micron rms poke [counts].

    double sciQuadGain{ 0 }; ///< rms of |H c| for a 1 micron rms poke; camsci1 adds |H c|^2 [counts]. 0 disables.

    double wfsNoise{ 2 }; ///< camwfs per-frame noise rms [counts].

    double sciNoise{ 2 }; ///< camsci1 per-frame noise rms [counts].

    double drift{ 0.001 }; ///< Fractional level drift per second, both cameras.

    unsigned seed{ 1 }; ///< Random seed for the response matrices.
};

/// Simulates dmcomb, camwfs, and camsci1 with known linear responses to the DM.
/** Each camera frame is `level*(1 + drift*t) + R * dm + noise`, where `dm` is the combined DM
 * stream. camsci1 optionally adds `|H dm|^2` with H complex (`sciQuadGain`), a stand-in for the
 * speckle intensity |S|^2. camsci1 frames average the DM state over the exposure, so a poke during an exposure
 * partially contaminates that frame, as on the instrument. camsci1 frames are posted
 * `sciPostDelay` after the exposure ends, so a poke made just after a post lands partway into
 * the next exposure, as with readout and shmimTCP latency.
 */
class xlinqSim
{
  protected:
    /** \name Configuration and Streams - Data
     * @{
     */

    xlinqSimConfig m_cfg; ///< The configuration.

    simStream m_dm; ///< The DM channel, float.

    simStream m_comb; ///< The combined DM stream, float.

    simStream m_wfs; ///< The camwfs stream, uint16.

    simStream m_sci; ///< The camsci1 stream, float.

    simStream m_modeval; ///< The optional modal-coefficient stream, float, written with each camwfs frame.

    mx::improc::eigenImage<float> m_modevalR; ///< Modal response matrix, nModes x nAct, coefficient per micron.

    mx::improc::eigenImage<float> m_wfsR; ///< camwfs response matrix, nWfsPix x nAct, counts per micron.

    mx::improc::eigenImage<float> m_sciR; ///< camsci1 response matrix, nSciPix x nAct, counts per micron.

    mx::improc::eigenImage<float> m_sciHr; ///< Real part of the camsci1 quadratic field response, nSciPix x nAct.

    mx::improc::eigenImage<float> m_sciHi; ///< Imaginary part of the camsci1 quadratic field response.

    ///@}

    /** \name Simulation State - Data
     * @{
     */

    std::mutex m_sigMutex; ///< Protects m_wfsSig and m_sciSig.

    mx::improc::eigenImage<float> m_wfsSig; ///< Current camwfs response to the combined DM, nWfsPix x 1.

    mx::improc::eigenImage<float> m_sciSig; ///< Current camsci1 response to the combined DM, nSciPix x 1.

    mx::improc::eigenImage<float> m_modevalSig; ///< Current modal coefficients of the combined DM, nModes x 1.

    std::atomic<bool> m_run{ false }; ///< True while the threads should run.

    std::thread m_combThread; ///< Combines the DM channel into the combined stream.

    std::thread m_wfsThread; ///< Produces camwfs frames.

    std::thread m_sciThread; ///< Produces camsci1 frames.

    std::chrono::steady_clock::time_point m_t0; ///< Start time, for drift.

    ///@}

  public:
    /// C'tor, validates names and builds the response matrices.
    /** \throws std::invalid_argument if a stream name lacks the required prefix
     */
    explicit xlinqSim( const xlinqSimConfig &cfg /**< [in] the configuration */ );

    /// D'tor, stops the threads and destroys the streams.
    ~xlinqSim();

    /// Create the streams and start the threads.
    void start();

    /// Stop the threads and destroy the streams.
    void stop();

    /// Get the camwfs response matrix, nWfsPix x nAct.
    const mx::improc::eigenImage<float> &wfsR() const;

    /// Get the camsci1 response matrix, nSciPix x nAct.
    const mx::improc::eigenImage<float> &sciR() const;

    /// Get the modal response matrix, nModes x nAct.
    const mx::improc::eigenImage<float> &modevalR() const;

    /// Get the real part of the camsci1 quadratic field response, nSciPix x nAct.
    const mx::improc::eigenImage<float> &sciHr() const;

    /// Get the imaginary part of the camsci1 quadratic field response, nSciPix x nAct.
    const mx::improc::eigenImage<float> &sciHi() const;

    /// Make a cube of random shapes, each normalized to 1 micron rms.
    static mx::improc::eigenCube<float> randomPokes( uint32_t rows, /**< [in] DM first dimension */
                                                     uint32_t cols, /**< [in] DM second dimension */
                                                     int      n,    /**< [in] number of shapes */
                                                     unsigned seed /**< [in] random seed */ );

  protected:
    /// The combiner thread: copy the channel to the combined stream after each channel post.
    void combThreadExec();

    /// The camwfs thread.
    void wfsThreadExec();

    /// The camsci1 thread.
    void sciThreadExec();

    /// Get seconds since start.
    double elapsed() const;

    /// True if the shutter file says the shutter is shut.
    bool shutterShut() const;

    /// The light transmission set by attenFile: attenFactor if it says On, else 1.
    double transmission() const;
};

inline simStream::~simStream()
{
    destroy();
}

inline void simStream::create( const std::string &name, uint8_t datatype, uint32_t rows, uint32_t cols )
{
    destroy();

    uint32_t size[2] = { rows, cols };

    if( ImageStreamIO_createIm_gpu(
            &m_image, name.c_str(), 2, size, datatype, -1, 1, IMAGE_NB_SEMAPHORE, 0, MATH_DATA, 0 ) !=
        IMAGESTREAMIO_SUCCESS )
    {
        throw std::runtime_error( "could not create stream " + name );
    }

    memset( m_image.array.raw, 0, rows * cols * ImageStreamIO_typesize( datatype ) );
    m_open = true;
}

inline void simStream::destroy()
{
    if( m_open )
    {
        ImageStreamIO_destroyIm( &m_image );
        m_open = false;
    }
}

inline IMAGE *simStream::image()
{
    return &m_image;
}

template <typename T>
T *simStream::data()
{
    return static_cast<T *>( m_image.array.raw );
}

inline void simStream::setWrite()
{
    m_image.md->write = 1;
}

inline void simStream::post()
{
    ImageStreamIO_UpdateIm( &m_image );
}

inline void simStream::post( const timespec &atime )
{
    timespec at = atime;
    ImageStreamIO_UpdateIm_atime( &m_image, &at );
}

inline xlinqSim::xlinqSim( const xlinqSimConfig &cfg ) : m_cfg( cfg )
{
    auto checkPrefix = [&cfg]( const std::string &n )
    {
        if( cfg.prefix.empty() || n.compare( 0, cfg.prefix.size(), cfg.prefix ) != 0 )
        {
            throw std::invalid_argument( "simulated stream " + n + " must start with prefix '" + cfg.prefix + "'" );
        }
    };

    for( const std::string &n : { cfg.dmChannel, cfg.dmComb, cfg.wfsName, cfg.sciName } )
    {
        checkPrefix( n );
    }

    if( !cfg.modevalName.empty() )
    {
        checkPrefix( cfg.modevalName );
    }

    size_t nAct = cfg.dmRows * cfg.dmCols;

    std::mt19937                    gen( cfg.seed );
    std::normal_distribution<float> norm( 0, 1 );

    m_wfsR.resize( cfg.wfsRows * cfg.wfsCols, nAct );
    m_sciR.resize( cfg.sciRows * cfg.sciCols, nAct );

    // Scaled so that a 1 micron rms shape gives a response with rms of about the gain
    float wfsScale = cfg.wfsGain / sqrt( nAct );
    float sciScale = cfg.sciGain / sqrt( nAct );

    m_wfsR = m_wfsR.unaryExpr(
        [&]( float )
        {
            return wfsScale * norm( gen );
        } );
    m_sciR = m_sciR.unaryExpr(
        [&]( float )
        {
            return sciScale * norm( gen );
        } );

    // Each of Hr and Hi gives sciQuadGain/sqrt(2) rms, so |H c| has rms sciQuadGain for a 1 micron rms shape
    float quadScale = cfg.sciQuadGain / sqrt( 2.0 * nAct );

    m_sciHr.resize( m_sciR.rows(), nAct );
    m_sciHi.resize( m_sciR.rows(), nAct );

    auto quadDraw = [&]( float )
    {
        return quadScale * norm( gen );
    };

    m_sciHr = m_sciHr.unaryExpr( quadDraw );
    m_sciHi = m_sciHi.unaryExpr( quadDraw );

    // Modal coefficients of order 1 for a 1 micron rms shape
    m_modevalR.resize( cfg.nModes, nAct );
    m_modevalR = m_modevalR.unaryExpr(
        [&]( float )
        {
            return norm( gen ) / static_cast<float>( sqrt( nAct ) );
        } );
    m_modevalSig.setZero( cfg.nModes, 1 );

    m_wfsSig.resize( m_wfsR.rows(), 1 );
    m_sciSig.resize( m_sciR.rows(), 1 );
    m_wfsSig.setZero();
    m_sciSig.setZero();
}

inline xlinqSim::~xlinqSim()
{
    stop();
}

inline void xlinqSim::start()
{
    m_dm.create( m_cfg.dmChannel, _DATATYPE_FLOAT, m_cfg.dmRows, m_cfg.dmCols );
    m_comb.create( m_cfg.dmComb, _DATATYPE_FLOAT, m_cfg.dmRows, m_cfg.dmCols );
    m_wfs.create( m_cfg.wfsName, _DATATYPE_UINT16, m_cfg.wfsRows, m_cfg.wfsCols );
    m_sci.create( m_cfg.sciName, _DATATYPE_FLOAT, m_cfg.sciRows, m_cfg.sciCols );

    if( !m_cfg.modevalName.empty() )
    {
        m_modeval.create( m_cfg.modevalName, _DATATYPE_FLOAT, m_cfg.nModes, 1 );
    }

    m_t0  = std::chrono::steady_clock::now();
    m_run = true;

    m_combThread = std::thread( &xlinqSim::combThreadExec, this );
    m_wfsThread  = std::thread( &xlinqSim::wfsThreadExec, this );
    m_sciThread  = std::thread( &xlinqSim::sciThreadExec, this );
}

inline void xlinqSim::stop()
{
    m_run = false;

    for( std::thread *t : { &m_combThread, &m_wfsThread, &m_sciThread } )
    {
        if( t->joinable() )
        {
            t->join();
        }
    }

    m_dm.destroy();
    m_comb.destroy();
    m_wfs.destroy();
    m_sci.destroy();
    m_modeval.destroy();
}

inline const mx::improc::eigenImage<float> &xlinqSim::wfsR() const
{
    return m_wfsR;
}

inline const mx::improc::eigenImage<float> &xlinqSim::sciR() const
{
    return m_sciR;
}

inline const mx::improc::eigenImage<float> &xlinqSim::modevalR() const
{
    return m_modevalR;
}

inline const mx::improc::eigenImage<float> &xlinqSim::sciHr() const
{
    return m_sciHr;
}

inline const mx::improc::eigenImage<float> &xlinqSim::sciHi() const
{
    return m_sciHi;
}

inline mx::improc::eigenCube<float> xlinqSim::randomPokes( uint32_t rows, uint32_t cols, int n, unsigned seed )
{
    std::mt19937                    gen( seed );
    std::normal_distribution<float> norm( 0, 1 );

    mx::improc::eigenCube<float> pokes( rows, cols, n );

    for( int k = 0; k < n; ++k )
    {
        auto im = pokes.image( k );
        im      = im.unaryExpr(
            [&]( float )
            {
                return norm( gen );
            } );
        im /= sqrt( im.square().sum() / im.size() );
    }

    return pokes;
}

inline void xlinqSim::combThreadExec()
{
    IMAGE *dm     = m_dm.image();
    int    semIdx = ImageStreamIO_getsemwaitindex( dm, 0 );

    size_t nAct = m_cfg.dmRows * m_cfg.dmCols;

    while( m_run )
    {
        timespec ts;
        clock_gettime( CLOCK_REALTIME, &ts );
        ts.tv_nsec += 100000000;

        if( ts.tv_nsec >= 1000000000 )
        {
            ts.tv_sec += 1;
            ts.tv_nsec -= 1000000000;
        }

        if( sem_timedwait( dm->semptr[semIdx], &ts ) != 0 )
        {
            continue;
        }

        std::this_thread::sleep_for( std::chrono::duration<double>( m_cfg.combDelay ) );

        Eigen::Map<Eigen::VectorXf> cmd( m_dm.data<float>(), nAct );

        m_comb.setWrite();
        Eigen::Map<Eigen::VectorXf>( m_comb.data<float>(), nAct ) = cmd;

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_sigMutex );
            m_wfsSig     = ( m_wfsR.matrix() * cmd ).array();
            m_modevalSig = ( m_modevalR.matrix() * cmd ).array();
            m_sciSig     = ( m_sciR.matrix() * cmd ).array();

            if( m_cfg.sciQuadGain > 0 )
            {
                m_sciSig += ( m_sciHr.matrix() * cmd ).array().square() + ( m_sciHi.matrix() * cmd ).array().square();
            }
        }

        m_comb.post();
    }

    dm->semReadPID[semIdx] = 0;
}

inline void xlinqSim::wfsThreadExec()
{
    std::mt19937                    gen( m_cfg.seed + 1 );
    std::normal_distribution<float> norm( 0, m_cfg.wfsNoise );

    size_t                        nPix = m_cfg.wfsRows * m_cfg.wfsCols;
    mx::improc::eigenImage<float> frame( nPix, 1 );
    mx::improc::eigenImage<float> modes( m_cfg.nModes, 1 );

    std::normal_distribution<float> modeNorm( 0, m_cfg.modevalNoise );

    auto period = std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        std::chrono::duration<double>( 1.0 / m_cfg.wfsFps ) );
    auto next = std::chrono::steady_clock::now();

    while( m_run )
    {
        next += period;
        std::this_thread::sleep_until( next );

        float level = m_cfg.wfsLevel * ( 1 + m_cfg.drift * elapsed() );

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_sigMutex );
            frame = level + m_wfsSig;
            modes = m_modevalSig;
        }

        frame = frame.unaryExpr(
            [&]( float v )
            {
                return v + norm( gen );
            } );

        m_wfs.setWrite();
        uint16_t *p = m_wfs.data<uint16_t>();

        for( size_t i = 0; i < nPix; ++i )
        {
            p[i] = static_cast<uint16_t>( std::clamp( std::lround( frame( i ) ), 0L, 65535L ) );
        }

        timespec wfsAtime = realtimeNow();
        m_wfs.post( wfsAtime );

        // The modal coefficients are computed from each camwfs frame, as in the loop
        if( !m_cfg.modevalName.empty() )
        {
            m_modeval.setWrite();
            Eigen::Map<Eigen::ArrayXf>( m_modeval.data<float>(), m_cfg.nModes ) = modes.unaryExpr(
                [&]( float v )
                {
                    return v + modeNorm( gen );
                } );
            m_modeval.post( wfsAtime ); // cacao carries the camera atime through
        }
    }
}

inline void xlinqSim::sciThreadExec()
{
    std::mt19937                    gen( m_cfg.seed + 2 );
    std::normal_distribution<float> norm( 0, m_cfg.sciNoise );

    size_t                        nPix = m_cfg.sciRows * m_cfg.sciCols;
    mx::improc::eigenImage<float> acc( nPix, 1 );
    mx::improc::eigenImage<float> pendingFrame( nPix, 1 );
    acc.setZero();

    // The PSF core, fixed
    mx::improc::eigenImage<float> core( m_cfg.sciRows, m_cfg.sciCols );
    core.setZero();
    if( m_cfg.sciCoreAmp > 0 )
    {
        double c0 = 0.5 * ( m_cfg.sciRows - 1 ), c1 = 0.5 * ( m_cfg.sciCols - 1 );
        for( uint32_t c = 0; c < m_cfg.sciCols; ++c )
        {
            for( uint32_t r = 0; r < m_cfg.sciRows; ++r )
            {
                double r2    = ( r - c0 ) * ( r - c0 ) + ( c - c1 ) * ( c - c1 );
                core( r, c ) = m_cfg.sciCoreAmp * exp( -0.5 * r2 / ( m_cfg.sciCoreSigma * m_cfg.sciCoreSigma ) );
            }
        }
    }
    Eigen::Map<const mx::improc::eigenImage<float>> coreVec( core.data(), nPix, 1 );

    typedef std::chrono::steady_clock::duration durT;

    durT sample    = std::chrono::duration_cast<durT>( std::chrono::duration<double>( 1.0 / m_cfg.sciSampleHz ) );
    durT expTime   = std::chrono::duration_cast<durT>( std::chrono::duration<double>( 1.0 / m_cfg.sciFps ) );
    durT postDelay = std::chrono::duration_cast<durT>( std::chrono::duration<double>( m_cfg.sciPostDelay ) );

    auto     expStart = std::chrono::steady_clock::now();
    auto     next     = expStart;
    auto     postAt   = expStart;
    bool     pending  = false;
    int      nSamp    = 0;
    timespec pendingAtime{};

    while( m_run )
    {
        // Average the DM response over the exposure
        { //mutex scope
            std::lock_guard<std::mutex> lock( m_sigMutex );
            acc += m_sciSig;
        }

        ++nSamp;
        next += sample;
        std::this_thread::sleep_until( next );

        // Post the previous frame after the readout/transfer delay, while the next exposure proceeds
        if( pending && next >= postAt )
        {
            m_sci.setWrite();
            Eigen::Map<Eigen::ArrayXf>( m_sci.data<float>(), nPix ) = pendingFrame;
            m_sci.post( pendingAtime ); // atime = end of exposure, as a camera timestamp
            pending = false;
        }

        if( next >= expStart + expTime )
        {
            float level = m_cfg.sciLevel * ( 1 + m_cfg.drift * ( elapsed() - 0.5 / m_cfg.sciFps ) );

            if( shutterShut() )
            {
                pendingFrame = pendingFrame.unaryExpr(
                    [&]( float )
                    {
                        return static_cast<float>( m_cfg.sciDark + norm( gen ) );
                    } );
            }
            else
            {
                float t = static_cast<float>( transmission() );
                pendingFrame =
                    ( m_cfg.sciDark + t * ( level - m_cfg.sciDark + acc / nSamp + coreVec ) )
                        .unaryExpr(
                            [&]( float v )
                            {
                                v += norm( gen );
                                return m_cfg.sciSat > 0 ? std::min( v, static_cast<float>( m_cfg.sciSat ) ) : v;
                            } );

                if( m_cfg.sciSat > 0 && m_cfg.sciBleed > 0 )
                {
                    const float sat = static_cast<float>( m_cfg.sciSat );
                    for( uint32_t c = 0; c < m_cfg.sciCols; ++c )
                    {
                        float *col  = pendingFrame.data() + c * m_cfg.sciRows;
                        int    last = -1;
                        for( uint32_t r = 0; r < m_cfg.sciRows; ++r )
                        {
                            if( col[r] >= sat )
                            {
                                last = r;
                            }
                        }
                        for( int r = last + 1; last >= 0 && r <= last + m_cfg.sciBleed && r < (int)m_cfg.sciRows; ++r )
                        {
                            col[r] = sat;
                        }
                    }
                }
            }
            pending      = true;
            pendingAtime = realtimeNow();
            postAt       = next + postDelay;
            expStart += expTime;
            acc.setZero();
            nSamp = 0;
        }
    }
}

inline bool xlinqSim::shutterShut() const
{
    if( m_cfg.shutterFile.empty() )
    {
        return false;
    }

    std::ifstream fin( m_cfg.shutterFile );
    std::string   state;
    fin >> state;

    return state.compare( 0, 2, "On" ) == 0;
}

inline double xlinqSim::transmission() const
{
    if( m_cfg.attenFile.empty() )
    {
        return 1;
    }

    std::ifstream fin( m_cfg.attenFile );
    std::string   state;
    fin >> state;

    return state.compare( 0, 2, "On" ) == 0 ? m_cfg.attenFactor : 1;
}

inline double xlinqSim::elapsed() const
{
    return std::chrono::duration<double>( std::chrono::steady_clock::now() - m_t0 ).count();
}

} // namespace xlinqCal

#endif // xlinqSim_hpp
