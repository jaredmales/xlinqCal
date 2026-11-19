/** \file isioStream.hpp
 * \brief An ImageStreamIO stream reader with semaphore handling and typed frame copies.
 *
 * \ingroup xlinqCal_files
 */

#ifndef isioStream_hpp
#define isioStream_hpp

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <semaphore.h>
#include <stdexcept>
#include <string>

#include <ImageStreamIO/ImageStreamIO.h>

#include <mx/improc/eigenImage.hpp>

namespace xlinqCal
{

/// The outcome of a timed semaphore wait.
enum class waitResult
{
    posted,  ///< The semaphore was posted.
    timeout, ///< The wait timed out or was interrupted by a signal.
    error    ///< sem_timedwait failed for another reason.
};

/// An ImageStreamIO stream opened for reading, optionally with its own claimed semaphore.
/** The semaphore is claimed with `ImageStreamIO_getsemwaitindex` and released on close by clearing
 * its `semReadPID` entry, following the pattern of MagAOX `dev::dmPokeWFS`. Note that ImageStreamIO
 * assigns at most one semaphore per process to a given stream.
 */
class isioStream
{
  protected:
    /** \name Stream State - Data
     * @{
     */

    std::string m_name; ///< The stream name, without the `.im.shm` extension.

    IMAGE m_image{}; ///< The ImageStreamIO image structure, valid while m_open is true.

    bool m_open{ false }; ///< True if the stream is open.

    int m_semIndex{ -1 }; ///< The claimed semaphore index, or -1 if none is claimed.

    ///@}

  public:
    /// Default c'tor.
    isioStream() = default;

    /// Copying is not allowed since the stream mapping is owned.
    isioStream( const isioStream & ) = delete;

    /// Copying is not allowed since the stream mapping is owned.
    isioStream &operator=( const isioStream & ) = delete;

    /// D'tor, closes the stream and releases the semaphore.
    ~isioStream();

    /// Open the stream and optionally claim a semaphore.
    /** \throws std::runtime_error if the stream cannot be opened or no semaphore is available.
     */
    void open( const std::string &name,    /**< [in] the stream name, without `.im.shm` */
               bool               claimSem /**< [in] [optional] whether to claim a semaphore for waiting */
               = true );

    /// Release the semaphore and close the stream.
    void close();

    /** \name Stream State
     * @{
     */

    /// Get whether the stream is open.
    bool isOpen() const;

    /// Get the stream name.
    const std::string &name() const;

    /// Get a pointer to the IMAGE structure.
    IMAGE *image();

    /// Get the first dimension of a frame.
    uint32_t rows() const;

    /// Get the second dimension of a frame.
    uint32_t cols() const;

    /// Get the ImageStreamIO datatype code.
    uint8_t datatype() const;

    /// Get the frame counter.
    uint64_t cnt0() const;

    /// Get the write time of the latest frame.
    timespec writetime() const;

    /// Get the claimed semaphore index, -1 if none.
    int semIndex() const;

    ///@}

    /// Flush the claimed semaphore so the next wait returns on the next post.
    void flush();

    /// Wait on the claimed semaphore.
    /** \returns waitResult::posted if posted
     * \returns waitResult::timeout on timeout or EINTR
     * \returns waitResult::error otherwise, including if no semaphore is claimed
     */
    waitResult wait( double timeout /**< [in] the timeout in seconds */ );

    /// Copy the current frame, converting to realT.
    /** The frame counter is read before and after the copy. If it changes the frame was
     * overwritten during the copy, and false is returned.
     *
     * \returns true if the copy is complete and untorn
     * \returns false if the frame changed during the copy
     *
     * \throws std::runtime_error for unsupported datatypes
     */
    template <typename realT>
    bool copy( mx::improc::eigenImage<realT> &im, /**< [out] the frame, resized as needed */
               uint64_t                      &cnt0 /**< [out] the frame counter of the copied frame */ );

    /// Copy a typed source array into a realT image.
    /** \throws std::runtime_error for unsupported datatypes
     */
    template <typename realT>
    static void copyTyped( mx::improc::eigenImage<realT> &im,       /**< [out] the destination, already sized */
                           const void                    *src,      /**< [in] the source pixels */
                           uint8_t                        datatype, /**< [in] the ImageStreamIO datatype code */
                           size_t                         rows,     /**< [in] the first dimension */
                           size_t                         cols /**< [in] the second dimension */ );
};

inline isioStream::~isioStream()
{
    close();
}

inline void isioStream::open( const std::string &name, bool claimSem )
{
    close();

    if( ImageStreamIO_openIm( &m_image, name.c_str() ) != IMAGESTREAMIO_SUCCESS )
    {
        throw std::runtime_error( "could not open stream " + name );
    }

    m_name = name;
    m_open = true;

    if( claimSem )
    {
        m_semIndex = ImageStreamIO_getsemwaitindex( &m_image, 0 );

        if( m_semIndex < 0 )
        {
            close();
            throw std::runtime_error( "no free semaphore on stream " + name );
        }

        flush();
    }
}

inline void isioStream::close()
{
    if( !m_open )
    {
        return;
    }

    if( m_semIndex >= 0 )
    {
        m_image.semReadPID[m_semIndex] = 0;
        m_semIndex                     = -1;
    }

    ImageStreamIO_closeIm( &m_image );
    m_open = false;
}

inline bool isioStream::isOpen() const
{
    return m_open;
}

inline const std::string &isioStream::name() const
{
    return m_name;
}

inline IMAGE *isioStream::image()
{
    return &m_image;
}

inline uint32_t isioStream::rows() const
{
    return m_image.md->size[0];
}

inline uint32_t isioStream::cols() const
{
    return m_image.md->naxis > 1 ? m_image.md->size[1] : 1;
}

inline uint8_t isioStream::datatype() const
{
    return m_image.md->datatype;
}

inline uint64_t isioStream::cnt0() const
{
    return m_image.md->cnt0;
}

inline timespec isioStream::writetime() const
{
    return m_image.md->writetime;
}

inline int isioStream::semIndex() const
{
    return m_semIndex;
}

inline void isioStream::flush()
{
    if( m_semIndex >= 0 )
    {
        ImageStreamIO_semflush( &m_image, m_semIndex );
    }
}

inline waitResult isioStream::wait( double timeout )
{
    if( m_semIndex < 0 )
    {
        return waitResult::error;
    }

    timespec ts;
    clock_gettime( CLOCK_REALTIME, &ts );

    double whole = std::floor( timeout );
    ts.tv_sec += static_cast<time_t>( whole );
    ts.tv_nsec += static_cast<long>( ( timeout - whole ) * 1e9 );

    if( ts.tv_nsec >= 1000000000 )
    {
        ts.tv_sec += 1;
        ts.tv_nsec -= 1000000000;
    }

    if( sem_timedwait( m_image.semptr[m_semIndex], &ts ) == 0 )
    {
        return waitResult::posted;
    }

    if( errno == ETIMEDOUT || errno == EINTR )
    {
        return waitResult::timeout;
    }

    return waitResult::error;
}

template <typename realT>
bool isioStream::copy( mx::improc::eigenImage<realT> &im, uint64_t &cnt0 )
{
    size_t nr = rows();
    size_t nc = cols();

    if( im.rows() != static_cast<long>( nr ) || im.cols() != static_cast<long>( nc ) )
    {
        im.resize( nr, nc );
    }

    cnt0 = m_image.md->cnt0;

    size_t slice = 0;
    if( m_image.md->naxis == 3 && m_image.md->size[2] > 1 )
    {
        slice = m_image.md->cnt1;
    }

    const char *src = static_cast<const char *>( m_image.array.raw ) +
                      slice * nr * nc * ImageStreamIO_typesize( m_image.md->datatype );

    copyTyped( im, src, m_image.md->datatype, nr, nc );

    return cnt0 == m_image.md->cnt0;
}

template <typename realT>
void isioStream::copyTyped(
    mx::improc::eigenImage<realT> &im, const void *src, uint8_t datatype, size_t rows, size_t cols )
{
    auto conv = [&]( auto *p )
    {
        typedef std::remove_const_t<std::remove_pointer_t<decltype( p )>> srcT;
        im = Eigen::Map<const Eigen::Array<srcT, -1, -1>>( p, rows, cols ).template cast<realT>();
    };

    switch( datatype )
    {
    case _DATATYPE_UINT8:
        conv( static_cast<const uint8_t *>( src ) );
        break;
    case _DATATYPE_INT8:
        conv( static_cast<const int8_t *>( src ) );
        break;
    case _DATATYPE_UINT16:
        conv( static_cast<const uint16_t *>( src ) );
        break;
    case _DATATYPE_INT16:
        conv( static_cast<const int16_t *>( src ) );
        break;
    case _DATATYPE_UINT32:
        conv( static_cast<const uint32_t *>( src ) );
        break;
    case _DATATYPE_INT32:
        conv( static_cast<const int32_t *>( src ) );
        break;
    case _DATATYPE_UINT64:
        conv( static_cast<const uint64_t *>( src ) );
        break;
    case _DATATYPE_INT64:
        conv( static_cast<const int64_t *>( src ) );
        break;
    case _DATATYPE_FLOAT:
        conv( static_cast<const float *>( src ) );
        break;
    case _DATATYPE_DOUBLE:
        conv( static_cast<const double *>( src ) );
        break;
    default:
        throw std::runtime_error( "unsupported ImageStreamIO datatype " + std::to_string( datatype ) );
    }
}

} // namespace xlinqCal

#endif // isioStream_hpp
