-- Run this once in the Supabase SQL Editor.
create table if not exists public.analytics_events (
    id bigint generated always as identity primary key,
    event_time timestamptz not null default now(),
    visitor_hash text not null,
    session_id uuid not null,
    event_type text not null check (
        event_type in ('visit', 'single_prediction', 'batch_prediction')
    ),
    item_count integer not null default 0 check (item_count >= 0),
    country_code text,
    country_name text,
    region text,
    latitude double precision,
    longitude double precision
);

create index if not exists analytics_events_time_idx
    on public.analytics_events (event_time);

create index if not exists analytics_events_type_idx
    on public.analytics_events (event_type);

create index if not exists analytics_events_visitor_idx
    on public.analytics_events (visitor_hash);

alter table public.analytics_events enable row level security;

-- No public RLS policies are needed. The Streamlit backend uses a Supabase
-- secret key stored only in Streamlit secrets. Never commit that key.
