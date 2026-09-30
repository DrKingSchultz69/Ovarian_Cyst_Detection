"use client";

import { useEffect, useRef, useState } from "react";

/** Run an async function when its dependencies change, with cancellation.
 *
 *  WHY THIS EXISTS, beyond tidiness. Four pages fetch from the API when a
 *  control changes -- the registration rotation, the phantom seed, the codec. The
 *  naive effect has a race: change the seed three times quickly and three
 *  requests are in flight, so the one that RESOLVES last wins rather than the
 *  one requested last. On a 3.5-second endpoint that is easy to hit, and the
 *  symptom is a page showing data for a seed the control no longer displays.
 *
 *  Each run takes a ticket. A result is only committed if its ticket is still
 *  the current one, so a superseded response is dropped rather than painted.
 *  The same guard covers unmount, which is the ordinary React warning about
 *  setting state on a gone component.
 *
 *  `react-hooks/set-state-in-effect` is disabled once, here, instead of at
 *  every call site. The rule is warning about cascading renders from
 *  synchronous setState in an effect body; this is the "subscribe to an
 *  external system" case the rule's own documentation exempts -- an HTTP API
 *  is precisely such a system -- and the loading flag has to be set before the
 *  request starts or the spinner never appears.
 */
export interface AsyncState<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

export function useAsync<T>(
  fn: () => Promise<T>,
  deps: readonly unknown[],
  options: { enabled?: boolean } = {},
): AsyncState<T> & { reload: () => void } {
  const enabled = options.enabled ?? true;

  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    error: null,
    loading: enabled,
  });

  // Bumped to force a re-run without changing the real dependencies.
  const [nonce, setNonce] = useState(0);

  // Latest ticket. A run whose ticket is stale must not commit.
  const ticket = useRef(0);

  // Held in a ref so the fetch effect does not re-run merely because the
  // caller passed a fresh closure -- re-running is controlled by `deps` alone.
  //
  // The assignment lives in its own effect rather than in the render body:
  // mutating a ref during render is a real violation (it breaks under
  // concurrent rendering, where a render can be thrown away). Declaration
  // order matters here -- this effect is declared first, so it has already
  // refreshed the ref by the time the fetch effect below runs.
  const fnRef = useRef(fn);
  useEffect(() => {
    fnRef.current = fn;
  });

  useEffect(() => {
    if (!enabled) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setState({ data: null, error: null, loading: false });
      return;
    }

    const mine = ++ticket.current;
    // No disable needed here: the updater form does not trip
    // react-hooks/set-state-in-effect, because the linter cannot prove it
    // changes anything. The reset above uses the object form and does.
    setState((prev) => ({ ...prev, loading: true, error: null }));

    void (async () => {
      try {
        const data = await fnRef.current();
        if (ticket.current === mine) setState({ data, error: null, loading: false });
      } catch (e) {
        if (ticket.current === mine) {
          setState({
            data: null,
            error: e instanceof Error ? e.message : String(e),
            loading: false,
          });
        }
      }
    })();

    // Invalidate this run's ticket on cleanup, so an in-flight response from a
    // superseded dependency change is discarded instead of overwriting the
    // newer one.
    return () => {
      if (ticket.current === mine) ticket.current += 1;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, nonce]);

  return { ...state, reload: () => setNonce((n) => n + 1) };
}
