/** \file indiClient.hpp
 * \brief A minimal INDI client that shells out to the MagAO-X getINDI and setINDI tools.
 *
 * \ingroup xlinqCal_files
 */

#ifndef indiClient_hpp
#define indiClient_hpp

#include <chrono>
#include <csignal>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

namespace xlinqCal
{

/// The result of running an external command.
struct cmdResult
{
    int status{ -1 }; ///< The exit status, or -1 if it did not exit normally.

    bool timedOut{ false }; ///< True if the command was killed at the timeout.

    std::string out; ///< Standard output.

    std::string err; ///< Standard error.
};

/// Run a command without a shell, capturing its output, and kill it at a timeout.
/** \returns the result; status is -1 if the command could not be started, was killed, or crashed
 */
inline cmdResult runCommand( const std::vector<std::string> &argv, /**< [in] the program and its arguments */
                             double                          timeout /**< [in] the wall-clock limit [s] */ );

/// A minimal INDI client using the MagAO-X getINDI and setINDI tools.
/** Follows the MagAOX rules for INDI clients (see planning/initial_plan.md):
 * - every get is one exact `device.property`, which makes getINDI send a named getProperties and exit
 *   as soon as the property arrives (no wildcards, no device-wide requests, no dumps of all properties);
 * - every set is type-flagged (`-s` or `-n`), so setINDI sends only the new value;
 * - every call has a wall-clock timeout; callers poll no faster than every 2 s.
 *
 * Names are restricted to [A-Za-z0-9_-] components, and commands run without a shell.
 */
class indiClient
{
  protected:
    std::string m_getCmd{ "getINDI" }; ///< The getINDI command.

    std::string m_setCmd{ "setINDI" }; ///< The setINDI command.

    double m_timeout{ 8 }; ///< Wall-clock limit for each call [s].

    int m_getWait{ 2 }; ///< getINDI -t value [s], its wait for the property.

  public:
    /// Set the commands.
    void commands( const std::string &getCmd, /**< [in] the getINDI command */
                   const std::string &setCmd /**< [in] the setINDI command */ );

    /// Set the wall-clock limit for each call [s].
    void timeout( double t /**< [in] the limit */ );

    /// Check a `device.property` (nDots = 1) or `device.property.element` (nDots = 2) name.
    static bool validName( const std::string &name, /**< [in] the name */
                           int                nDots /**< [in] the number of components minus one */ );

    /// Get one property.
    /** \returns 0 on success, with elems mapping element name to value and raw holding getINDI's lines
     * \returns -1 on error (invalid name, timeout, not found, or no elements)
     */
    int getProperty( const std::string                  &devProp, /**< [in] `device.property` */
                     std::map<std::string, std::string> &elems,   /**< [out] element name to value */
                     std::string                        &raw,     /**< [out] the getINDI output lines */
                     std::string                        &errMsg /**< [out] the error, if any */ );

    /// Get one element.
    /** \returns 0 on success, -1 on error
     */
    int getElement( const std::string &devPropElem, /**< [in] `device.property.element` */
                    std::string       &value,       /**< [out] the value */
                    std::string       &errMsg /**< [out] the error, if any */ );

    /// Set a switch element On or Off with `setINDI -s`.
    /** setINDI exits 0 once the request is sent; callers confirm by reading back.
     * \returns 0 if sent, -1 on error
     */
    int setSwitch( const std::string &devPropElem, /**< [in] `device.property.element` */
                   bool               on,          /**< [in] true for On */
                   std::string       &errMsg /**< [out] the error, if any */ );
};

inline cmdResult runCommand( const std::vector<std::string> &argv, double timeout )
{
    cmdResult res;

    int outPipe[2];
    int errPipe[2];

    if( argv.empty() || pipe( outPipe ) != 0 )
    {
        return res;
    }

    if( pipe( errPipe ) != 0 )
    {
        close( outPipe[0] );
        close( outPipe[1] );
        return res;
    }

    pid_t pid = fork();

    if( pid < 0 )
    {
        for( int fd : { outPipe[0], outPipe[1], errPipe[0], errPipe[1] } )
        {
            close( fd );
        }
        return res;
    }

    if( pid == 0 )
    {
        // Child: default signal handling, stdout/stderr to the pipes, stdin from /dev/null
        signal( SIGINT, SIG_DFL );
        signal( SIGTERM, SIG_DFL );

        dup2( outPipe[1], STDOUT_FILENO );
        dup2( errPipe[1], STDERR_FILENO );

        int devNull = open( "/dev/null", O_RDONLY );
        if( devNull >= 0 )
        {
            dup2( devNull, STDIN_FILENO );
        }

        std::vector<char *> args;
        for( const std::string &a : argv )
        {
            args.push_back( const_cast<char *>( a.c_str() ) );
        }
        args.push_back( nullptr );

        execvp( args[0], args.data() );
        _exit( 127 );
    }

    close( outPipe[1] );
    close( errPipe[1] );

    auto deadline = std::chrono::steady_clock::now() + std::chrono::duration<double>( timeout );

    pollfd fds[2] = { { outPipe[0], POLLIN, 0 }, { errPipe[0], POLLIN, 0 } };
    int    nOpen  = 2;
    char   buf[4096];

    while( nOpen > 0 )
    {
        int ms = std::chrono::duration_cast<std::chrono::milliseconds>( deadline - std::chrono::steady_clock::now() )
                     .count();

        if( ms <= 0 )
        {
            res.timedOut = true;
            kill( pid, SIGKILL );
            break;
        }

        int pr = poll( fds, 2, ms );

        if( pr < 0 )
        {
            if( errno == EINTR )
            {
                continue;
            }
            kill( pid, SIGKILL );
            break;
        }

        for( int i = 0; i < 2; ++i )
        {
            if( fds[i].fd >= 0 && ( fds[i].revents & ( POLLIN | POLLHUP | POLLERR ) ) )
            {
                ssize_t n = read( fds[i].fd, buf, sizeof( buf ) );

                if( n > 0 )
                {
                    ( i == 0 ? res.out : res.err ).append( buf, n );
                }
                else if( n == 0 || errno != EINTR )
                {
                    close( fds[i].fd );
                    fds[i].fd = -1;
                    --nOpen;
                }
            }
        }
    }

    for( pollfd &p : fds )
    {
        if( p.fd >= 0 )
        {
            close( p.fd );
        }
    }

    int wstatus = 0;
    while( waitpid( pid, &wstatus, 0 ) < 0 && errno == EINTR )
    {
    }

    if( !res.timedOut && WIFEXITED( wstatus ) )
    {
        res.status = WEXITSTATUS( wstatus );
    }

    return res;
}

inline void indiClient::commands( const std::string &getCmd, const std::string &setCmd )
{
    m_getCmd = getCmd;
    m_setCmd = setCmd;
}

inline void indiClient::timeout( double t )
{
    m_timeout = t;
}

inline bool indiClient::validName( const std::string &name, int nDots )
{
    int  dots  = 0;
    bool empty = true;

    for( char c : name )
    {
        if( c == '.' )
        {
            if( empty )
            {
                return false;
            }
            ++dots;
            empty = true;
        }
        else if( isalnum( static_cast<unsigned char>( c ) ) || c == '_' || c == '-' )
        {
            empty = false;
        }
        else
        {
            return false;
        }
    }

    return dots == nDots && !empty;
}

inline int indiClient::getProperty( const std::string                  &devProp,
                                    std::map<std::string, std::string> &elems,
                                    std::string                        &raw,
                                    std::string                        &errMsg )
{
    elems.clear();
    raw.clear();

    if( !validName( devProp, 1 ) )
    {
        errMsg = "invalid INDI property name '" + devProp + "' (expected device.property)";
        return -1;
    }

    cmdResult r = runCommand( { m_getCmd, "-t", std::to_string( m_getWait ), devProp }, m_timeout );

    if( r.timedOut || r.status != 0 )
    {
        errMsg = m_getCmd + " " + devProp + ( r.timedOut ? " timed out" : " failed" ) +
                 ( r.err.empty() ? "" : ": " + r.err.substr( 0, r.err.find( '\n' ) ) );
        return -1;
    }

    std::istringstream is( r.out );
    std::string        line;
    std::string        prefix = devProp + ".";

    while( std::getline( is, line ) )
    {
        size_t eq = line.find( '=' );

        if( eq == std::string::npos || line.compare( 0, prefix.size(), prefix ) != 0 )
        {
            continue;
        }

        elems[line.substr( prefix.size(), eq - prefix.size() )] = line.substr( eq + 1 );
        raw += line + "\n";
    }

    if( elems.empty() )
    {
        errMsg = m_getCmd + " " + devProp + " returned no elements";
        return -1;
    }

    return 0;
}

inline int indiClient::getElement( const std::string &devPropElem, std::string &value, std::string &errMsg )
{
    if( !validName( devPropElem, 2 ) )
    {
        errMsg = "invalid INDI element name '" + devPropElem + "' (expected device.property.element)";
        return -1;
    }

    size_t                             dot = devPropElem.rfind( '.' );
    std::map<std::string, std::string> elems;
    std::string                        raw;

    if( getProperty( devPropElem.substr( 0, dot ), elems, raw, errMsg ) < 0 )
    {
        return -1;
    }

    auto it = elems.find( devPropElem.substr( dot + 1 ) );

    if( it == elems.end() )
    {
        errMsg = "no element " + devPropElem;
        return -1;
    }

    value = it->second;

    return 0;
}

inline int indiClient::setSwitch( const std::string &devPropElem, bool on, std::string &errMsg )
{
    if( !validName( devPropElem, 2 ) )
    {
        errMsg = "invalid INDI element name '" + devPropElem + "' (expected device.property.element)";
        return -1;
    }

    cmdResult r = runCommand( { m_setCmd, "-s", devPropElem + "=" + ( on ? "On" : "Off" ) }, m_timeout );

    if( r.timedOut || r.status != 0 )
    {
        errMsg = m_setCmd + " " + devPropElem + ( r.timedOut ? " timed out" : " failed" ) +
                 ( r.err.empty() ? "" : ": " + r.err.substr( 0, r.err.find( '\n' ) ) );
        return -1;
    }

    return 0;
}

} // namespace xlinqCal

#endif // indiClient_hpp
