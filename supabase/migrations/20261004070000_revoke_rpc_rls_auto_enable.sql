-- Security advisor 0028/0029 (2026-10-04): public.rls_auto_enable() is the SECURITY DEFINER function behind Supabase's
-- `ensure_rls` event trigger. Event triggers are fired by the server, so no client role needs EXECUTE; revoking it
-- removes the /rest/v1/rpc/rls_auto_enable surface without changing what the trigger does.
revoke execute on function public.rls_auto_enable() from public, anon, authenticated;
