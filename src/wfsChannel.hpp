/** \file wfsChannel.hpp
 * \brief A WFS stream read by its own thread and accumulated while a gate is open, plus per-camera products.
 *
 * \ingroup xlinqCal_files
 */

#ifndef wfsChannel_hpp
#define wfsChannel_hpp

#include <atomic>
#include <mutex>
#include <string>
#include <thread>

#include <mx/improc/eigenCube.hpp>
#include <mx/improc/eigenImage.hpp>

#include "isioStream.hpp"

namespace xlinqCal
{

/// The per-poke products accumulated for one camera or WFS stream.
struct camProducts
{
    mx::improc::eigenCube<float> resp; ///< Odd part (mean(+) - mean(-))/2, one plane per poke.

    mx::improc::eigenCube<float> mean; ///< Symmetric mean (mean(+) + mean(-))/2, one plane per poke.

    mx::improc::eigenCube<float> ref; ///< Zero-poke references, in acquisition order.

    /// Allocate and zero the cubes.
    void allocate( uint32_t rows,  /**< [in] the first dimension */
                   uint32_t cols,  /**< [in] the second dimension */
                   int      nPoke, /**< [in] planes for resp and mean */
                   int      nRef /**< [in] planes for ref, 0 for none */ );
};

/// Frame statistics for one gate of a WFS channel.
struct wfsGateStats
{
    uint64_t n{ 0 }; ///< Number of frames accumulated.

    uint64_t missed{ 0 }; ///< cnt0 gaps while the gate was open.

    uint64_t torn{ 0 }; ///< Frames dropped because they changed during the copy.

    uint64_t cnt0First{ 0 }; ///< cnt0 of the first accumulated frame.

    uint64_t cnt0Last{ 0 }; ///< cnt0 of the last accumulated frame.

    double atimeFirst{ 0 }; ///< Acquisition time of the first accumulated frame, Unix seconds.

    double atimeLast{ 0 }; ///< Acquisition time of the last accumulated frame, Unix seconds.
};

/// A WFS stream (camera frames, calibrated frames, or modal coefficients) read by its own thread.
/** The thread waits on the stream's semaphore and copies each new frame once. While the gate is
 * open, frames are summed and counted. Repeated posts for an already-seen frame are ignored, cnt0
 * gaps are counted as missed, and frames that change during the copy are dropped as torn.
 */
class wfsChannel
{
  protected:
    /** \name Stream and Thread - Data
     * State shared with the reader thread is protected by m_mutex.
     * @{
     */

    isioStream m_stream; ///< The stream, with claimed semaphore.

    std::thread m_thread; ///< The reader thread.

    std::atomic<bool> m_run{ false }; ///< True while the reader thread should run.

    std::atomic<int> m_errno{ 0 }; ///< Nonzero if the reader thread exited on an error.

    std::mutex m_mutex; ///< Protects the gate, sum, and statistics.

    bool m_gate{ false }; ///< When true, frames are accumulated.

    bool m_gateFirst{ false }; ///< True until the first frame after the gate opens, for gap counting.

    mx::improc::eigenImage<double> m_sum; ///< Sum of frames while the gate is open.

    wfsGateStats m_stats; ///< Statistics for the current gate.

    uint64_t m_lastSeen{ 0 }; ///< cnt0 of the last frame seen by the reader thread.

    ///@}

    /** \name Products - Data
     * @{
     */

    camProducts m_products; ///< The per-poke products for this stream.

    double m_fps{ 0 }; ///< Frame rate measured during preflight.

    ///@}

  public:
    /// Default c'tor.
    wfsChannel() = default;

    /// Copying is not allowed since the stream and thread are owned.
    wfsChannel( const wfsChannel & ) = delete;

    /// Copying is not allowed since the stream and thread are owned.
    wfsChannel &operator=( const wfsChannel & ) = delete;

    /// D'tor, stops the reader thread.
    ~wfsChannel();

    /// Open the stream and claim a semaphore.
    /** \throws std::runtime_error on error
     */
    void open( const std::string &name /**< [in] the stream name */ );

    /// Get the stream.
    isioStream &stream();

    /// Get the stream (const).
    const isioStream &stream() const;

    /// Get the per-poke products.
    camProducts &products();

    /// Get the per-poke products (const).
    const camProducts &products() const;

    /// Get the measured frame rate.
    double fps() const;

    /// Set the measured frame rate.
    void fps( double f /**< [in] the frame rate */ );

    /// Start the reader thread.
    void start();

    /// Stop and join the reader thread.
    void stop();

    /// Get the reader thread's error, 0 if none.
    int error() const;

    /// Zero the sum and statistics, and open the gate.
    void openGate();

    /// Close the gate and return the mean and statistics.
    /** \returns true if at least one frame was accumulated (mean is set)
     * \returns false if no frames were accumulated (mean is unchanged)
     */
    bool closeGate( mx::improc::eigenImage<double> &mean, /**< [out] the mean of the gated frames */
                    wfsGateStats                   &stats /**< [out] the gate statistics */ );

  protected:
    /// The reader thread entry point.
    void exec();
};

inline void camProducts::allocate( uint32_t rows, uint32_t cols, int nPoke, int nRef )
{
    for( mx::improc::eigenCube<float> *c : { &resp, &mean } )
    {
        c->resize( rows, cols, nPoke );
        c->setZero();
    }

    if( nRef > 0 )
    {
        ref.resize( rows, cols, nRef );
        ref.setZero();
    }
}

inline wfsChannel::~wfsChannel()
{
    stop();
}

inline void wfsChannel::open( const std::string &name )
{
    m_stream.open( name );
}

inline isioStream &wfsChannel::stream()
{
    return m_stream;
}

inline const isioStream &wfsChannel::stream() const
{
    return m_stream;
}

inline camProducts &wfsChannel::products()
{
    return m_products;
}

inline const camProducts &wfsChannel::products() const
{
    return m_products;
}

inline double wfsChannel::fps() const
{
    return m_fps;
}

inline void wfsChannel::fps( double f )
{
    m_fps = f;
}

inline void wfsChannel::start()
{
    m_sum.resize( m_stream.rows(), m_stream.cols() );
    m_sum.setZero();
    m_gate     = false;
    m_errno    = 0;
    m_lastSeen = m_stream.cnt0();
    m_run      = true;
    m_thread   = std::thread( &wfsChannel::exec, this );
}

inline void wfsChannel::stop()
{
    m_run = false;

    if( m_thread.joinable() )
    {
        m_thread.join();
    }
}

inline int wfsChannel::error() const
{
    return m_errno;
}

inline void wfsChannel::openGate()
{
    std::lock_guard<std::mutex> lock( m_mutex );

    m_sum.setZero();
    m_stats     = wfsGateStats();
    m_gateFirst = true;
    m_gate      = true;
}

inline bool wfsChannel::closeGate( mx::improc::eigenImage<double> &mean, wfsGateStats &stats )
{
    std::lock_guard<std::mutex> lock( m_mutex );

    m_gate = false;
    stats  = m_stats;

    if( m_stats.n == 0 )
    {
        return false;
    }

    mean = m_sum / static_cast<double>( m_stats.n );

    return true;
}

inline void wfsChannel::exec()
{
    mx::improc::eigenImage<double> frame;
    timespec                       atime{};

    while( m_run )
    {
        waitResult wr = m_stream.wait( 0.5 );

        if( wr == waitResult::timeout )
        {
            continue;
        }

        if( wr == waitResult::error )
        {
            m_errno = errno ? errno : -1;
            return;
        }

        uint64_t cnt0;
        bool     ok = m_stream.copy( frame, cnt0, atime );

        if( cnt0 == m_lastSeen )
        {
            continue; // A repeated post for a frame already seen
        }

        { //mutex scope
            std::lock_guard<std::mutex> lock( m_mutex );

            if( m_gate )
            {
                if( !m_gateFirst && cnt0 > m_lastSeen + 1 )
                {
                    m_stats.missed += cnt0 - m_lastSeen - 1;
                }

                m_gateFirst = false;

                if( ok )
                {
                    if( m_stats.n == 0 )
                    {
                        m_stats.cnt0First  = cnt0;
                        m_stats.atimeFirst = tsSec( atime );
                    }

                    m_sum += frame;
                    m_stats.cnt0Last  = cnt0;
                    m_stats.atimeLast = tsSec( atime );
                    ++m_stats.n;
                }
                else
                {
                    ++m_stats.torn;
                }
            }
        }

        m_lastSeen = cnt0;
    }
}

} // namespace xlinqCal

#endif // wfsChannel_hpp
